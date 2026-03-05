# -*- coding: utf-8 -*-
"""
生成交集棱柱网格

- 输入：一批行程（起点O/终点D、起始时间、行程时间等）
- 棱柱：每个行程的时空棱柱由 `生成mesh.get_prism_cache()` 产生/缓存（注意：棱柱构造使用的是 max_trip_time）
- 顺序：对每一对行程 (i, j)，分别检查两种接单顺序 O1→O2、O2→O1
- 等待时间：若存在到站时间差异，则将实际等待时间按各自乘客的最大等待上限截断后继续计算，不因超限直接丢弃该顺序
- 过滤：用 PostGIS `ST_3DIntersects` 做保守快速过滤
- 精算：用 PyMeshLab 布尔交集得到网格并落盘，同时保存 JSON 元数据

重要参数（与论文符号对应）：
- detour ratio/cap：额外行驶时间上限，用于 max_trip_time = trip_time + min(cap, ratio*trip_time)
- wait ratio/cap：最大等待时间上限，用于等待可行性判定
"""
import os
import csv
import shutil
import time
import numpy as np
import pymeshlab
import trimesh
import math
from typing import List, Dict, Optional, Tuple, Callable, TypedDict, Any
import pandas as pd
import pickle
import json

try:
    from shapely import speedups
except ImportError:
    speedups = None
from collections import Counter, defaultdict
import multiprocessing
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import psycopg2
from psycopg2 import pool

# 复用已有的棱柱缓存与mesh生成工具
from 生成mesh import get_prism_cache, PRISM_CACHE_DIR

# ========== 数据库配置 ==========
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'prism',
    'user': 'postgres',
    'password': 'SQL123'
}
ROAD_TABLE = 'Manhatton_road_copy2'

# 节点表名常量
VERTICES_TABLE = 'Manhatton_road_vertices_pgr2'

# 主进程数据库连接池（用于主进程的批量查询）
_main_db_pool = None

# 棱柱缓存实例（单例）
_prism_cache = None

def get_available_prism_trip_ids() -> set[int]:
    """
    扫描棱柱缓存目录，返回已经生成棱柱文件的 trip_id 集合。

    只在进入 ST_3DIntersects 之前调用一次，用于剪掉那些肯定没有棱柱文件的行程，
    避免它们进入 worker 里频繁触发 I/O / 加载失败。
    """
    trip_ids: set[int] = set()
    try:
        if not os.path.isdir(PRISM_CACHE_DIR):
            print(f"[预筛选] 棱柱缓存目录不存在或不可访问: {PRISM_CACHE_DIR}")
            return trip_ids

        for name in os.listdir(PRISM_CACHE_DIR):
            # 约定文件名格式：trip_{trip_id}.ply
            if not (name.startswith("trip_") and name.endswith(".ply")):
                continue
            num_str = name[len("trip_"):-len(".ply")]
            try:
                trip_ids.add(int(num_str))
            except Exception:
                # 文件名不符合规范时直接忽略
                continue
    except Exception as e:
        print(f"[预筛选] 扫描棱柱缓存目录失败: {e}")

    print(f"[预筛选] 检测到 {len(trip_ids)} 个已有棱柱的行程（trip_id）")
    return trip_ids


# 进程池 worker 全局状态类型定义
class WorkerStateDict(TypedDict):
    initialized: bool
    db_pool: Optional[pool.ThreadedConnectionPool]  # 每个worker进程内部的连接池


# 进程池 worker 全局状态（每个worker进程独立）
WORKER_STATE: WorkerStateDict = {
    'initialized': False,
    'db_pool': None  # 每个worker进程内部的连接池
}

# TT 输出目录
TT_OUTPUT_DIR = r'D:\shirou\carpool\分步结果数据\测试结果\节点_TTmesh_1375704000'
# 检查点目录（默认在TT输出目录的父目录下）
CHECKPOINT_DIR = os.path.join(os.path.dirname(TT_OUTPUT_DIR), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========== 配置常量 ==========
EVALUATION_WORKERS = 20  # 20核进程池
MAX_EXTRA_SECS = 300  # 兼容旧常量名（秒）；实际以 DETOUR_* 常量为准
max_inflight = EVALUATION_WORKERS * 3  # 最大并发任务数
CHECKPOINT_SAVE_INTERVAL = 1000000  # 每处理多少个任务保存一次检查点

# ========== 论文参数（建议只改这里/或用CLI覆盖） ==========
# 额外行驶时间（detour）上限：max_trip_time = trip_time + min(DETOUR_CAP_SECS, DETOUR_RATIO * trip_time)
DETOUR_RATIO = 0.15
DETOUR_CAP_SECS = 300.0
# 最大等待时间上限：max_wait_time_i = min(WAIT_CAP_SECS, WAIT_RATIO * trip_time_i)
WAIT_RATIO = 0.3
WAIT_CAP_SECS = 300.0

# 预筛选用的等效速度上限（25英里/小时 ≈ 11.18 m/s）
MAX_SPEED_MPS = 25.0 * 1609.34 / 3600.0

# 批量最近节点查询 chunk（避免 SQL 参数过长）
NEAREST_NODE_BATCH_SIZE = 2000


def get_main_db_connection_pool():
    """获取主进程的数据库连接池"""
    global _main_db_pool
    if _main_db_pool is None:
        _main_db_pool = pool.ThreadedConnectionPool(
            minconn=8,
            maxconn=24,
            **DB_CONFIG
        )
    return _main_db_pool


def reset_db_connection_pool():
    """重置主进程数据库连接池，确保下次重新建立连接"""
    global _main_db_pool
    if _main_db_pool is not None:
        try:
            _main_db_pool.closeall()
        except Exception:
            pass
        _main_db_pool = None


class CheckpointManager:
    """检查点管理器，用于保存和恢复处理进度"""

    def __init__(self, checkpoint_dir: str, checkpoint_name: str = "checkpoint"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")
        self.metadata_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_metadata.json")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, stage: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, merge: bool = True):
        """
        保存检查点（在已有检查点基础上合并新数据）

        Args:
            stage: 当前阶段名称（如 'candidate', 'st3d', 'tt'）
            data: 要保存的数据字典，包含当前阶段的中间结果（只需传入当前阶段的数据）
            metadata: 可选的元数据（如处理时间、进度等）
            merge: 是否合并模式（默认True）。如果True，会先加载已有检查点，然后合并新数据；
                   如果False，则完全覆盖（覆盖模式）
        """
        try:
            # 合并模式：先加载已有检查点
            existing_data = {}
            existing_metadata = {}
            if merge and os.path.exists(self.checkpoint_path):
                try:
                    with open(self.checkpoint_path, 'rb') as f:
                        existing_checkpoint = pickle.load(f)
                        existing_data = existing_checkpoint.get('data', {})
                        existing_metadata = existing_checkpoint.get('metadata', {})
                except Exception as e:
                    print(f"[检查点] 警告：加载已有检查点失败，将使用覆盖模式: {e}")
                    existing_data = {}
                    existing_metadata = {}

            # 合并数据：新数据会覆盖同名的旧数据，但保留其他阶段的数据
            merged_data = {**existing_data, **data}
            merged_metadata = {**existing_metadata, **(metadata or {})}

            checkpoint_data = {
                'stage': stage,
                'data': merged_data,
                'timestamp': time.time(),
                'metadata': merged_metadata
            }

            # 保存主数据（使用pickle以支持numpy数组等复杂对象）
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            # 保存元数据（使用JSON）
            metadata_json = {
                'stage': stage,
                'timestamp': checkpoint_data['timestamp'],
                'timestamp_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_data['timestamp'])),
                **checkpoint_data['metadata']
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_json, f, indent=2, ensure_ascii=False)

            mode_str = "合并" if merge else "覆盖"
            print(f"[检查点] 已保存阶段 '{stage}' 的检查点（{mode_str}模式）: {self.checkpoint_path}")
        except Exception as e:
            print(f"[检查点] 保存失败: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """
        加载检查点

        Returns:
            如果存在检查点，返回包含 'stage' 和 'data' 的字典；否则返回None
        """
        if not os.path.exists(self.checkpoint_path):
            return None

        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            print(f"[检查点] 从检查点恢复: 阶段='{checkpoint_data['stage']}', "
                  f"时间={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_data['timestamp']))}")
            return checkpoint_data
        except Exception as e:
            print(f"[检查点] 加载失败: {e}")
            return None

    def exists(self) -> bool:
        """检查检查点是否存在"""
        return os.path.exists(self.checkpoint_path)

    def clear(self):
        """清除检查点"""
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            print(f"[检查点] 已清除检查点")
        except Exception as e:
            print(f"[检查点] 清除失败: {e}")


def get_main_db_connection(max_retries=3, retry_delay=1):
    """获取主进程的数据库连接（使用连接池）"""
    for attempt in range(max_retries):
        try:
            pool_obj = get_main_db_connection_pool()
            conn = pool_obj.getconn()
            conn.autocommit = True
            return conn
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                conn.autocommit = True
                return conn
            except Exception as e2:
                if attempt == max_retries - 1:
                    raise


def return_main_db_connection(conn):
    """归还主进程的数据库连接到连接池"""
    try:
        if conn is None or conn.closed:
            return
        pool_obj = get_main_db_connection_pool()
        pool_obj.putconn(conn)
    except Exception:
        try:
            conn.close()
        except:
            pass


def _get_worker_db_pool():
    """
    获取worker进程内部的数据库连接池（每个worker进程独立）
    """
    global WORKER_STATE
    pool_obj = WORKER_STATE.get('db_pool')
    if pool_obj is None:
        WORKER_STATE['db_pool'] = pool.ThreadedConnectionPool(
            minconn=2,  # 每个worker进程最小连接数
            maxconn=8,  # 每个worker进程最大连接数
            **DB_CONFIG
        )
    return WORKER_STATE['db_pool']


def _worker_initializer():
    """
    进程池 initializer：确保每个 worker 只做一次昂贵初始化
    包括数据库连接池、Shapely speedups等。
    """
    global WORKER_STATE
    if WORKER_STATE.get('initialized'):
        return

    # 初始化连接池（每个worker进程独立）
    _get_worker_db_pool()

    # 启用Shapely speedups加速（如果可用）
    if speedups is not None:
        try:
            speedups.enable()
        except Exception:
            pass

    WORKER_STATE['initialized'] = True


def get_multiple_nearest_nodes_with_coords_batch(conn, points_list):
    """
    批量查找多个点的最近节点（参数化SQL + KNN，论文复现更稳健）。

    Args:
        conn: 数据库连接
        points_list: 点列表，格式为 [(lon, lat), ...]

    Returns:
        节点列表，格式为 [(node_id, (lon, lat)), ...]
    """
    if not points_list:
        return []

    results: List[Tuple[Optional[int], Optional[Tuple[float, float]]]] = [(None, None)] * len(points_list)

    def _coerce_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # 分块，避免 VALUES 过大
    with conn.cursor() as cur:
        for base in range(0, len(points_list), NEAREST_NODE_BATCH_SIZE):
            chunk = points_list[base:base + NEAREST_NODE_BATCH_SIZE]
            values_sql_parts = []
            params = []
            for offset, (lon, lat) in enumerate(chunk):
                lon_f = _coerce_float(lon)
                lat_f = _coerce_float(lat)
                idx = base + offset
                # 即便 lon/lat 无法转换，也保留占位，查询会返回空，结果保持 (None, None)
                values_sql_parts.append("(%s, %s, %s)")
                params.extend([idx, lon_f, lat_f])

            values_sql = ",".join(values_sql_parts)
            # 使用 <-> KNN 运算符（需要 GIST/SPGIST 索引支持，通常 vertices 表会有）
            sql = f"""
            WITH pts(idx, lon, lat) AS (
                VALUES {values_sql}
            )
            SELECT
                pts.idx,
                v.id,
                ST_X(v.the_geom) AS x,
                ST_Y(v.the_geom) AS y
            FROM pts
            LEFT JOIN LATERAL (
                SELECT id, the_geom
                FROM {VERTICES_TABLE}
                WHERE pts.lon IS NOT NULL AND pts.lat IS NOT NULL
                ORDER BY the_geom <-> ST_SetSRID(ST_Point(pts.lon, pts.lat), 4326)
                LIMIT 1
            ) AS v ON TRUE
            ORDER BY pts.idx
            """
            cur.execute(sql, params)
            rows = cur.fetchall()
            for idx, node_id, x, y in rows:
                if idx is None or node_id is None or x is None or y is None:
                    continue
                try:
                    results[int(idx)] = (int(node_id), (float(x), float(y)))
                except Exception:
                    # 保守：保持 None
                    continue

    return results


def _compute_max_wait_time(trip: Dict) -> float:
    """
    根据单个行程的 trip_time 计算最大等待时间
    """
    trip_time = float(trip.get('trip_time', 0.0) or 0.0)
    return min(WAIT_RATIO * trip_time, WAIT_CAP_SECS)


def _time_intervals_overlap_with_wait(trip1: Dict, trip2: Dict) -> bool:
    """
    使用 [start_time, start_time + max_trip_time + max_wait_time] 的时间轴区间做快速重叠检测。

    只要区间完全不重叠，则在时间和等待约束下基本不可能产生有效拼车，
    可在进入 ST_3DIntersects 之前直接剪枝。
    """
    start1 = float(trip1.get('start_time', 0.0) or 0.0)
    start2 = float(trip2.get('start_time', 0.0) or 0.0)

    trip_time1 = float(trip1.get('trip_time', 0.0) or 0.0)
    trip_time2 = float(trip2.get('trip_time', 0.0) or 0.0)
    max_trip_time1 = float(trip1.get('max_trip_time', trip_time1) or trip_time1)
    max_trip_time2 = float(trip2.get('max_trip_time', trip_time2) or trip_time2)

    max_wait_time1 = _compute_max_wait_time(trip1)
    max_wait_time2 = _compute_max_wait_time(trip2)

    end1 = start1 + max_trip_time1 + max_wait_time1
    end2 = start2 + max_trip_time2 + max_wait_time2

    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)

    return latest_start <= earliest_end


def _lonlat_to_xy(lon: float, lat: float, ref_lat: float) -> Tuple[float, float]:
    """
    将经纬度近似投影到局部平面坐标系（米），用于几何距离计算。

    采用简单的等距近似：
    - y 方向使用纬度弧长（R * dlat）
    - x 方向使用在参考纬度 ref_lat 处的经度弧长（R * cos(ref_lat) * dlon）
    """
    try:
        lon_f = float(lon)
        lat_f = float(lat)
    except Exception:
        return 0.0, 0.0

    R = 6371000.0  # 地球半径，单位：米
    ref_lat_rad = math.radians(ref_lat)
    x = math.radians(lon_f) * math.cos(ref_lat_rad) * R
    y = math.radians(lat_f) * R
    return x, y


def _point_to_segment_distance_m(p_xy: Tuple[float, float],
                                 a_xy: Tuple[float, float],
                                 b_xy: Tuple[float, float]) -> float:
    """
    计算平面上点到线段的最短距离（米）。
    """
    px, py = p_xy
    ax, ay = a_xy
    bx, by = b_xy

    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 0.0:
        # 退化成点
        return math.hypot(wx, wy)

    t = (wx * vx + wy * vy) / seg_len2
    if t <= 0.0:
        dx = wx
        dy = wy
    elif t >= 1.0:
        dx = px - bx
        dy = py - by
    else:
        proj_x = ax + t * vx
        proj_y = ay + t * vy
        dx = px - proj_x
        dy = py - proj_y

    return math.hypot(dx, dy)


def _o2_near_o1d1_segment(trip1: Dict, trip2: Dict) -> bool:
    """
    距离过滤：判断行程2的起点 O2 是否在行程1路径（O1→D1）的指定距离/时间缓冲带内。

    实现上用 O1 和 D1 的直线连线近似实际行驶路径 O1→D1，
    使用局部平面近似计算 O2 到线段 O1D1 的最短欧氏距离 dist_m，
    再用等效速度上限 MAX_SPEED_MPS 将距离转成时间：t_near = dist_m / MAX_SPEED_MPS。

    若 t_near <= max_trip_time1，则认为 O2 足够“靠近”行程1路径，保留该 pair；
    否则在几何上太远，可在进入 ST_3DIntersects 之前剪枝。

    为了安全起见，如果坐标缺失或 max_trip_time1 非正数，将保守返回 True（不过滤）。
    """
    O1 = trip1.get('O')
    D1 = trip1.get('D')
    O2 = trip2.get('O')
    if not O1 or not D1 or not O2:
        # 坐标信息不完整时保守不过滤
        return True

    try:
        lon1, lat1 = float(O1[0]), float(O1[1])
        lon_d1, lat_d1 = float(D1[0]), float(D1[1])
        lon2, lat2 = float(O2[0]), float(O2[1])
    except Exception:
        # 坐标解析失败时保守不过滤
        return True

    trip_time1 = float(trip1.get('trip_time', 0.0) or 0.0)
    max_trip_time1 = float(trip1.get('max_trip_time', trip_time1) or trip_time1)
    if max_trip_time1 <= 0.0 or MAX_SPEED_MPS <= 0.0:
        # 行程时间异常或速度上限异常时保守不过滤
        return True

    # 使用三点平均纬度作为局部投影的参考纬度
    ref_lat = (lat1 + lat_d1 + lat2) / 3.0

    a_xy = _lonlat_to_xy(lon1, lat1, ref_lat)
    b_xy = _lonlat_to_xy(lon_d1, lat_d1, ref_lat)
    p_xy = _lonlat_to_xy(lon2, lat2, ref_lat)

    dist_m = _point_to_segment_distance_m(p_xy, a_xy, b_xy)
    t_near = dist_m / MAX_SPEED_MPS

    return t_near <= max_trip_time1


def _o1_near_o2d2_segment(trip1: Dict, trip2: Dict) -> bool:
    """
    距离过滤（对称版）：判断行程1的起点 O1 是否在行程2路径（O2→D2）的指定距离/时间缓冲带内。

    实现与 `_o2_near_o1d1_segment` 对称：
    - 用 O2 和 D2 的直线连线近似实际行驶路径 O2→D2
    - 计算 O1 到线段 O2D2 的最短欧氏距离 dist_m
    - 使用等效速度上限 MAX_SPEED_MPS 将距离转成时间：t_near = dist_m / MAX_SPEED_MPS

    若 t_near <= max_trip_time2，则认为 O1 足够“靠近”行程2路径，保留该 pair；
    为保证安全，如果坐标或时间参数异常，则保守返回 True（不做过滤）。
    """
    O1 = trip1.get('O')
    O2 = trip2.get('O')
    D2 = trip2.get('D')
    if not O1 or not O2 or not D2:
        # 坐标信息不完整时保守不过滤
        return True

    try:
        lon1, lat1 = float(O1[0]), float(O1[1])
        lon2, lat2 = float(O2[0]), float(O2[1])
        lon_d2, lat_d2 = float(D2[0]), float(D2[1])
    except Exception:
        # 坐标解析失败时保守不过滤
        return True

    trip_time2 = float(trip2.get('trip_time', 0.0) or 0.0)
    max_trip_time2 = float(trip2.get('max_trip_time', trip_time2) or trip_time2)
    if max_trip_time2 <= 0.0 or MAX_SPEED_MPS <= 0.0:
        # 行程时间异常或速度上限异常时保守不过滤
        return True

    # 使用三点平均纬度作为局部投影的参考纬度
    ref_lat = (lat1 + lat2 + lat_d2) / 3.0

    a_xy = _lonlat_to_xy(lon2, lat2, ref_lat)
    b_xy = _lonlat_to_xy(lon_d2, lat_d2, ref_lat)
    p_xy = _lonlat_to_xy(lon1, lat1, ref_lat)

    dist_m = _point_to_segment_distance_m(p_xy, a_xy, b_xy)
    t_near = dist_m / MAX_SPEED_MPS

    return t_near <= max_trip_time2


def start_monitoring():
    """
    启动性能监控（占位函数，需要根据实际需求实现）。

    Returns:
        监控对象，包含 record_stage_start 和 record_stage_end 方法
    """

    class Monitor:
        def __init__(self):
            self.stages = {}

        def record_stage_start(self, stage_name):
            self.stages[stage_name] = {'start': time.time()}

        def record_stage_end(self, stage_name):
            if stage_name in self.stages:
                self.stages[stage_name]['end'] = time.time()
                elapsed = self.stages[stage_name]['end'] - self.stages[stage_name]['start']
                print(f"[监控] {stage_name} 完成，耗时: {elapsed:.2f}秒")

    return Monitor()


def _get_worker_connection(max_retries=3, retry_delay=1):
    """
    从worker进程内部的连接池获取数据库连接，如失效则自动重连。
    """
    for attempt in range(max_retries):
        try:
            pool = _get_worker_db_pool()
            conn = pool.getconn()
            conn.autocommit = True
            return conn
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            # 如果连接池失败，尝试直接连接
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                conn.autocommit = True
                return conn
            except Exception as e2:
                if attempt == max_retries - 1:
                    raise


def _return_worker_connection(conn):
    """
    归还连接到worker进程内部的连接池
    """
    try:
        if conn is None:
            return
        if conn.closed:
            return
        pool = _get_worker_db_pool()
        pool.putconn(conn)
    except Exception:
        try:
            conn.close()
        except:
            pass


def repair_mesh(mesh: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
    """尽量不改变原始数据的网格修复"""
    try:
        mesh.process(True)  # 确保顶点顺序
        mesh.process(validate=False)  # 禁用顶点合并
        mesh.update_faces(mesh.unique_faces())  # 移除重复三角面
        mesh.update_faces(mesh.nondegenerate_faces())  # 去除退化三角形面
        mesh.remove_infinite_values()  # 删除包含 inf、-inf 或 NaN 的顶点
        mesh.fix_normals(multibody=True)

        if not mesh.is_watertight:
            mesh.fill_holes()
            if not mesh.is_watertight and not mesh.is_volume:
                return None
        return mesh
    except Exception as e:
        print(f"网格修复出错: {str(e)}")
        return None


def mesh_to_postgis_geometry(mesh: trimesh.Trimesh) -> Optional[str]:
    """
    将trimesh转换为PostGIS 3D TIN格式的WKT字符串。
    TIN (Triangulated Irregular Network) 格式更适合三角形mesh。

    根据PostGIS文档，TIN格式为：
    TIN Z (((x1 y1 z1, x2 y2 z2, x3 y3 z3, x1 y1 z1)), ((x4 y4 z4, x5 y5 z5, x6 y6 z6, x4 y4 z4)), ...)
    注意：每个三角形必须是闭合的，需要4个点（第一个点重复一次以形成闭合环）

    Args:
        mesh: trimesh.Trimesh对象

    Returns:
        PostGIS TIN Z格式的WKT字符串，如果转换失败则返回None
    """
    try:
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None

        # 获取顶点和面
        vertices = mesh.vertices  # (N, 3) array: [x, y, z]
        faces = mesh.faces  # (M, 3) array: [i, j, k]

        # 构建TIN Z的WKT格式
        # TIN Z格式：每个三角形必须是闭合的，需要4个点（第一个点重复一次）
        # 格式：((x1 y1 z1, x2 y2 z2, x3 y3 z3, x1 y1 z1))
        triangles = []
        for face in faces:
            # 获取面的三个顶点索引
            v1_idx, v2_idx, v3_idx = int(face[0]), int(face[1]), int(face[2])

            # 确保索引有效
            if v1_idx >= len(vertices) or v2_idx >= len(vertices) or v3_idx >= len(vertices):
                continue

            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            v3 = vertices[v3_idx]

            # TIN格式：每个三角形必须是闭合的，需要4个点（第一个点重复一次）
            # 格式：((x1 y1 z1, x2 y2 z2, x3 y3 z3, x1 y1 z1))
            coords = f"{v1[0]} {v1[1]} {v1[2]}, {v2[0]} {v2[1]} {v2[2]}, {v3[0]} {v3[1]} {v3[2]}, {v1[0]} {v1[1]} {v1[2]}"
            triangles.append(f"(({coords}))")

        if not triangles:
            return None

        # 组合成TIN Z格式
        # 格式：TIN Z (((x1 y1 z1, x2 y2 z2, x3 y3 z3)), ((x4 y4 z4, x5 y5 z5, x6 y6 z6)), ...)
        wkt = f"TIN Z ({','.join(triangles)})"
        return wkt
    except Exception as e:
        print(f"mesh转换为PostGIS geometry失败: {str(e)}")
        return None


def compute_mesh_intersection(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
    """
    计算两个网格的交集，只返回交集mesh（不落盘）。
    """
    try:
        # 修复与凸包兜底
        if not mesh1.is_watertight:
            mesh1 = repair_mesh(mesh1)
            if mesh1 is None or not mesh1.is_watertight:
                mesh1 = mesh1.convex_hull if mesh1 is not None else None
        if not mesh2.is_watertight:
            mesh2 = repair_mesh(mesh2)
            if mesh2 is None or not mesh2.is_watertight:
                mesh2 = mesh2.convex_hull if mesh2 is not None else None
        if mesh1 is None or mesh2 is None:
            return None

        # 预处理网格
        mesh1.update_faces(mesh1.unique_faces())
        mesh2.update_faces(mesh2.unique_faces())
        mesh1.fix_normals()
        mesh2.fix_normals()
        mesh1.process(validate=True)
        mesh2.process(validate=True)
        mesh1.fill_holes()
        mesh2.fill_holes()

        # 使用 PyMeshLab 布尔交集
        temp_dir = tempfile.gettempdir()
        tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".ply", dir=temp_dir)
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".ply", dir=temp_dir)
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".ply", dir=temp_dir)
        tmp1.close()
        tmp2.close()
        tmp_out.close()
        try:
            mesh1.export(tmp1.name)
            mesh2.export(tmp2.name)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(tmp1.name)
            ms.load_new_mesh(tmp2.name)
            ms.mesh_boolean_intersection(first_mesh=0, second_mesh=1)
            result_mesh = ms.current_mesh()
            if result_mesh.vertex_number() == 0 or result_mesh.face_number() == 0:
                return None
            ms.save_current_mesh(tmp_out.name)
            intersection_mesh = trimesh.load(tmp_out.name)
        finally:
            for path in (tmp1.name, tmp2.name, tmp_out.name):
                try:
                    os.remove(path)
                except OSError:
                    pass

        if intersection_mesh is None or intersection_mesh.is_empty:
            return None

        intersection_mesh = repair_mesh(intersection_mesh)
        if intersection_mesh is None:
            return None
        if intersection_mesh.is_watertight and getattr(intersection_mesh, "is_volume", True):
            return intersection_mesh

        # 凸包兜底
        intersection_mesh = intersection_mesh.convex_hull
        intersection_mesh = repair_mesh(intersection_mesh)
        return intersection_mesh if intersection_mesh and intersection_mesh.is_watertight else None
    except Exception as e:
        # print(f"计算网格交集时出错: {str(e)}")
        return None


def check_mesh_3d_intersects(conn, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> bool:
    """
    使用ST_3DIntersects快速判断两个mesh是否有交集。

    Args:
        conn: 数据库连接
        mesh1: 第一个mesh对象
        mesh2: 第二个mesh对象

    Returns:
        True表示有交集，False表示无交集
    """
    try:
        # 将mesh转换为PostGIS TIN格式的WKT字符串
        geom1_wkt = mesh_to_postgis_geometry(mesh1)
        geom2_wkt = mesh_to_postgis_geometry(mesh2)

        if geom1_wkt is None or geom2_wkt is None:
            # 转换失败，保守返回True（不跳过后续计算）
            return True

        # 使用ST_3DIntersects进行快速判断
        # PostGIS会自动使用边界框比较进行快速过滤
        sql = """
        SELECT ST_3DIntersects(
            ST_GeomFromText(%s, 4326),
            ST_GeomFromText(%s, 4326)
        )
        """
        with conn.cursor() as cur:
            cur.execute(sql, (geom1_wkt, geom2_wkt))
            result = cur.fetchone()
            if result and result[0] is not None:
                return bool(result[0])
            # 如果查询结果为空或None，保守返回True
            return True
    except Exception as e:
        # 异常时保守返回True（不跳过后续计算）
        # print(f"ST_3DIntersects检查失败: {str(e)}")
        return True


def get_nearest_node_single(conn, lon: float, lat: float):
    """
    单点最近路网节点查询（TT 阶段专用，极轻量）
    返回: (node_id: int, (lon: float, lat: float)) 或 (None, None)
    """
    sql = f"""
    SELECT id, ST_X(the_geom), ST_Y(the_geom)
    FROM {VERTICES_TABLE}
    ORDER BY the_geom <-> ST_SetSRID(ST_Point(%s, %s), 4326)
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (lon, lat))
        row = cur.fetchone()
        if row:
            # 确保node_id是int类型，处理numpy数组、Decimal等类型
            node_id_raw = row[0]
            if node_id_raw is not None:
                try:
                    if hasattr(node_id_raw, 'item'):
                        # numpy标量（如numpy.int64）
                        node_id = int(node_id_raw.item())
                    elif isinstance(node_id_raw, (list, tuple, np.ndarray)):
                        # 如果是数组，取第一个元素
                        node_id = int(node_id_raw[0]) if len(node_id_raw) > 0 else None
                    else:
                        # Decimal、int等类型，直接转换为int
                        node_id = int(node_id_raw)
                except (ValueError, TypeError, IndexError) as e:
                    print(
                        f"警告：get_nearest_node_single中node_id类型转换失败，类型={type(node_id_raw)}, 值={node_id_raw}, 错误={e}")
                    node_id = None
            else:
                node_id = None

            # 确保坐标是float类型
            node_lon = float(row[1]) if row[1] is not None else None
            node_lat = float(row[2]) if row[2] is not None else None

            if node_id is not None and node_lon is not None and node_lat is not None:
                return node_id, (node_lon, node_lat)
            else:
                return None, None
        return None, None


def get_shortest_travel_time(conn, origin_node_id: int, destination_node_id: int) -> Optional[float]:
    """
    获取两个节点间的最短行驶时间（使用pgRouting）

    Args:
        conn: 数据库连接
        origin_node_id: 起点节点ID
        destination_node_id: 终点节点ID

    Returns:
        最短行驶时间（秒），如果无法到达返回None
    """
    try:
        with conn.cursor() as cur:
            sql = f"""
            SELECT agg_cost
            FROM pgr_dijkstra(
                'SELECT gid as id, source, target, forward_travel_time as cost, backward_travel_time as reverse_cost 
                 FROM {ROAD_TABLE}',
                {origin_node_id}, {destination_node_id}, true
            )
            WHERE node = {destination_node_id}
            LIMIT 1
            """
            cur.execute(sql)
            result = cur.fetchone()

            if result and result[0] is not None:
                return float(result[0])
            else:
                return None

    except Exception as e:
        # 静默处理错误，避免大量输出
        return None


def batch_get_shortest_travel_times(conn, node_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Optional[float]]:
    """
    批量获取多个节点对之间的最短行驶时间

    Args:
        conn: 数据库连接
        node_pairs: 节点对列表，格式为 [(origin_node_id, destination_node_id), ...]

    Returns:
        字典，键为节点对 (origin_node_id, destination_node_id)，值为最短行驶时间（秒），如果无法到达则为None
    """
    # 结果字典，默认全部为 None
    results: Dict[Tuple[int, int], Optional[float]] = {}
    if not node_pairs:
        return results

    # 先初始化所有 pair 的结果为 None
    for pair in node_pairs:
        results[pair] = None

    # 按 origin 分组：对每个 origin 只跑一次 one-to-many Dijkstra
    from collections import defaultdict

    origin_to_dests: Dict[int, List[int]] = defaultdict(list)
    for origin_id, dest_id in node_pairs:
        if origin_id is None or dest_id is None:
            # 保持为 None，跳过
            continue
        origin_to_dests[origin_id].append(dest_id)

    if not origin_to_dests:
        return results

    try:
        with conn.cursor() as cur:
            edges_sql = (
                f"SELECT gid as id, source, target, "
                f"forward_travel_time as cost, backward_travel_time as reverse_cost "
                f"FROM {ROAD_TABLE}"
            )

            for origin_id, dest_list in origin_to_dests.items():
                if not dest_list:
                    continue

                # 使用 pgRouting 的 one-to-many Dijkstra：
                # - start_vids: 单个起点 [origin_id]
                # - end_vids:   该起点对应的所有终点列表
                # 只保留 node 在目标集合中的行（即真正的 destination）
                sql = """
                SELECT node, agg_cost
                FROM pgr_dijkstra(
                    %s,      -- edges_sql
                    %s::int[],  -- start_vids
                    %s::int[],  -- end_vids
                    true     -- directed
                )
                WHERE node = ANY(%s::int[])
                """
                params = (edges_sql, [origin_id], dest_list, dest_list)
                try:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                except Exception:
                    # 当前 origin 的批量查询失败，退回到该 origin 的逐对查询
                    for dest_id in dest_list:
                        pair = (origin_id, dest_id)
                        results[pair] = get_shortest_travel_time(conn, origin_id, dest_id)
                    continue

                # 将本次 Dijkstra 的结果回填到 results
                # 可能有部分目的地不可达（没有返回行），保持为 None
                for node_id, cost in rows:
                    if node_id is None or cost is None:
                        continue
                    dest_id = int(node_id)
                    pair = (origin_id, dest_id)
                    if pair in results:
                        results[pair] = float(cost)

    except Exception:
        # 整体批量失败时，退回到完全逐对查询（保持原行为的兜底策略）
        for origin_id, dest_id in node_pairs:
            results[(origin_id, dest_id)] = get_shortest_travel_time(conn, origin_id, dest_id)

    return results


def _st3d_filter_worker(payload: Dict) -> Tuple[
    int, int, bool, float, bool, bool, bool, bool, str, str]:
    """
    ST_3DIntersects检查worker函数：判断两个行程的棱柱是否有交集。

    流程：
    加载两棱柱→时间对齐→ST_3DIntersects检查

    Args:
        payload: 包含所有必要信息的字典，包括：
            - i, j: 行程对索引
            - t1, t2: 两个行程的完整信息
            - base_time: 基准时间
            - o1_to_o2_time: O1到O2的最短驾驶时间（秒）
            - o2_to_o1_time: O2到O1的最短驾驶时间（秒）

    Returns:
        (i, j, has_intersect, elapsed_time, pass_o1o2, pass_o2o1, attempted_o1o2, attempted_o2o1, reason_o1o2, reason_o2o1)
        has_intersect: 只要任一顺序通过即为True
        pass_o1o2: True表示 O1→O2 顺序在考虑等待时间和平移后有交集
        pass_o2o1: True表示 O2→O1 顺序在考虑等待时间和平移后有交集
        attempted_o1o2/attempted_o2o1: 是否实际执行了 ST_3DIntersects（即满足行驶时间存在且等待时间不超限）
        reason_o1o2/reason_o2o1: 若未通过该顺序的原因（可能为
            'pass'/'missing_travel_time'/'wait_exceed'/'no_intersect'/'missing_mesh'/'mesh_load_failed'/'error'）
    """
    pair_start_time = time.perf_counter()
    i = payload['i']
    j = payload['j']
    t1 = payload['t1']
    t2 = payload['t2']
    base_time = payload['base_time']
    o1_to_o2_time = payload.get('o1_to_o2_time')
    o2_to_o1_time = payload.get('o2_to_o1_time')

    prism_cache = get_prism_cache()
    mesh_T1 = None
    mesh_T2 = None
    # 记录两种顺序是否通过
    pass_o1o2 = False
    pass_o2o1 = False
    attempted_o1o2 = False
    attempted_o2o1 = False
    reason_o1o2 = "no_intersect"  # 默认：若走到实际检查且不相交则保持该值
    reason_o2o1 = "no_intersect"
    trip_id1 = t1['trip_id']
    trip_id2 = t2['trip_id']

    try:
        # 1. 从缓存加载两个prism
        mesh_T1 = prism_cache.checkout_mesh(trip_id1)
        if mesh_T1 is None:
            prism_path1 = prism_cache.checkout(trip_id1)
            if not prism_path1:
                # mesh/prism 文件不存在：直接不通过 ST_3DIntersects（不再保守放行）
                try:
                    prism_cache.release(trip_id1)
                except Exception:
                    pass
                return (
                    i, j, False, time.perf_counter() - pair_start_time,
                    False, False, False, False, "missing_mesh", "missing_mesh"
                )
            # prism文件存在但加载失败：同样不通过（避免把坏数据放进 TT）
            prism_cache.release(trip_id1)
            return (
                i, j, False, time.perf_counter() - pair_start_time,
                False, False, False, False, "mesh_load_failed", "mesh_load_failed"
            )
        else:
            prism_cache.checkout(trip_id1)

        mesh_T2 = prism_cache.checkout_mesh(trip_id2)
        if mesh_T2 is None:
            prism_path2 = prism_cache.checkout(trip_id2)
            if not prism_path2:
                prism_cache.release(trip_id1)
                try:
                    prism_cache.release(trip_id2)
                except Exception:
                    pass
                return (
                    i, j, False, time.perf_counter() - pair_start_time,
                    False, False, False, False, "missing_mesh", "missing_mesh"
                )
            # prism文件存在但加载失败：不通过
            prism_cache.release(trip_id1)
            prism_cache.release(trip_id2)
            return (
                i, j, False, time.perf_counter() - pair_start_time,
                False, False, False, False, "mesh_load_failed", "mesh_load_failed"
            )
        else:
            prism_cache.checkout(trip_id2)

        if mesh_T1 is None or mesh_T2 is None:
            # 加载后为None：不通过
            prism_cache.release(trip_id1)
            prism_cache.release(trip_id2)
            return (
                i, j, False, time.perf_counter() - pair_start_time,
                False, False, False, False, "mesh_load_failed", "mesh_load_failed"
            )

        # 2. 计算基础时间偏移（对齐到 base_time）
        start_time1 = t1['start_time']
        start_time2 = t2['start_time']
        delta_t1 = start_time1 - base_time
        delta_t2 = start_time2 - base_time

        # 创建基础 mesh（对齐到 base_time）
        mesh_T1_base = mesh_T1.copy()
        mesh_T2_base = mesh_T2.copy()
        if delta_t1 != 0:
            mesh_T1_base.apply_translation([0, 0, delta_t1])
        if delta_t2 != 0:
            mesh_T2_base.apply_translation([0, 0, delta_t2])

        # 3. 计算最大容忍等待时间
        # 注意：棱柱构造使用 max_trip_time，因此用于时间对齐也应使用 max_trip_time
        trip_time1 = float(t1.get('trip_time', 0.0) or 0.0)
        trip_time2 = float(t2.get('trip_time', 0.0) or 0.0)
        max_trip_time1 = float(t1.get('max_trip_time', trip_time1) or trip_time1)
        max_trip_time2 = float(t2.get('max_trip_time', trip_time2) or trip_time2)

        max_wait_time1 = min(WAIT_RATIO * trip_time1, WAIT_CAP_SECS)
        max_wait_time2 = min(WAIT_RATIO * trip_time2, WAIT_CAP_SECS)

        conn = None
        try:
            conn = _get_worker_connection()

            # 4. 按 O1→O2 顺序进行等待场景判断与 ST_3DIntersects 筛选（等待超限截断到上限后继续）
            if o1_to_o2_time is None:
                reason_o1o2 = "missing_travel_time"
            else:
                vehicle_arrival_at_o2 = start_time1 + o1_to_o2_time
                diff_o1o2 = vehicle_arrival_at_o2 - start_time2  # >0 乘客2等车，<0 车（带乘客1）在等
                wait_time_o1o2 = abs(diff_o1o2)

                # 等待超限时截断到各自上限后继续：即平移量取“有符号的 min(实际等待, 上限)”
                if diff_o1o2 == 0:
                    wait_offset = 0.0
                else:
                    if diff_o1o2 < 0:
                        # 车（带乘客1）在等乘客2，受乘客1等待上限约束
                        capped_wait = min(wait_time_o1o2, max_wait_time1)
                    else:
                        # 乘客2在等车（带乘客1），受乘客2等待上限约束
                        capped_wait = min(wait_time_o1o2, max_wait_time2)
                    wait_offset = math.copysign(capped_wait, diff_o1o2)

                mesh_T2_translated = mesh_T2_base.copy()
                if wait_offset != 0:
                    mesh_T2_translated.apply_translation([0, 0, wait_offset])

                if mesh_T2_translated is not None:
                    attempted_o1o2 = True
                    if check_mesh_3d_intersects(conn, mesh_T1_base, mesh_T2_translated):
                        pass_o1o2 = True
                        reason_o1o2 = "pass"
                    else:
                        reason_o1o2 = "no_intersect"

            # 5. 按 O2→O1 顺序进行等待场景判断与 ST_3DIntersects 筛选（等待超限截断到上限后继续）
            if o2_to_o1_time is None:
                reason_o2o1 = "missing_travel_time"
            else:
                vehicle_arrival_at_o1 = start_time2 + o2_to_o1_time
                diff_o2o1 = vehicle_arrival_at_o1 - start_time1  # >0 乘客1等车，<0 车（带乘客2）在等
                wait_time_o2o1 = abs(diff_o2o1)

                if diff_o2o1 == 0:
                    wait_offset = 0.0
                else:
                    if diff_o2o1 < 0:
                        # 车（带乘客2）在等乘客1，受乘客2等待上限约束
                        capped_wait = min(wait_time_o2o1, max_wait_time2)
                    else:
                        # 乘客1在等车（带乘客2），受乘客1等待上限约束
                        capped_wait = min(wait_time_o2o1, max_wait_time1)
                    wait_offset = math.copysign(capped_wait, diff_o2o1)

                mesh_T1_translated = mesh_T1_base.copy()
                if wait_offset != 0:
                    mesh_T1_translated.apply_translation([0, 0, wait_offset])

                if mesh_T1_translated is not None:
                    attempted_o2o1 = True
                    if check_mesh_3d_intersects(conn, mesh_T1_translated, mesh_T2_base):
                        pass_o2o1 = True
                        reason_o2o1 = "pass"
                    else:
                        reason_o2o1 = "no_intersect"

            has_intersect = pass_o1o2 or pass_o2o1

        except Exception:
            # ST_3DIntersects 检查失败，保守放行并视为两种顺序都可能通过
            has_intersect = True
            pass_o1o2 = True
            pass_o2o1 = True
            attempted_o1o2 = True
            attempted_o2o1 = True
            reason_o1o2 = "error"
            reason_o2o1 = "error"
        finally:
            if conn is not None:
                _return_worker_connection(conn)

        # 6. 释放缓存引用并返回
        prism_cache.release(trip_id1)
        prism_cache.release(trip_id2)
        return (
            i,
            j,
            has_intersect,
            time.perf_counter() - pair_start_time,
            pass_o1o2,
            pass_o2o1,
            attempted_o1o2,
            attempted_o2o1,
            reason_o1o2,
            reason_o2o1,
        )
    except Exception as e:
        try:
            prism_cache.release(trip_id1)
            prism_cache.release(trip_id2)
        except:
            pass
        # 异常时保守返回True（不跳过后续计算），并视为两种顺序都可能通过
        return (
            i,
            j,
            True,
            time.perf_counter() - pair_start_time,
            True,
            True,
            True,
            True,
            "error",
            "error",
        )


def _evaluate_pair_worker(payload: Dict) -> Tuple[
    int, int, Optional[int], float, bool, Optional[str], Optional[str]]:
    """
    第五阶段worker函数：处理一对行程的完整TT计算流程（考虑等待时间的两种TT生成）。

    流程：
    1. 加载两棱柱
    2. 对两种TT顺序（O1→O2和O2→O1）分别处理：
       - 计算等待时间和最大容忍等待时间
       - 判断等待场景并平移mesh
       - ST_3DIntersects过滤
       - 如果通过，计算TT交集并保存

    Args:
        payload: 包含所有必要信息的字典，包括：
            - i, j: 行程对索引
            - t1, t2: 两个行程的完整信息
            - base_time: 基准时间
            - tt_dir: TT输出目录
            - o1_to_o2_time: O1到O2的最短驾驶时间（秒）
            - o2_to_o1_time: O2到O1的最短驾驶时间（秒）

    Returns:
        (i, j, tt_count, pair_time, success, failure_stage, failure_reason)
        tt_count: 成功生成的TT文件数量（0、1或2）
    """
    pair_start_time = time.perf_counter()
    i = payload['i']
    j = payload['j']
    t1 = payload['t1']
    t2 = payload['t2']
    base_time = payload['base_time']

    prism_cache = get_prism_cache()
    tt_dir = payload['tt_dir']
    o1_to_o2_time = payload.get('o1_to_o2_time')  # O1到O2的最短驾驶时间
    o2_to_o1_time = payload.get('o2_to_o1_time')  # O2到O1的最短驾驶时间
    # 来自 ST_3DIntersects 阶段 的顺序过滤结果，用于减少不必要的 pymeshlab 调用
    use_o1o2 = payload.get('use_o1o2', True)
    use_o2o1 = payload.get('use_o2o1', True)

    mesh_T1 = None
    mesh_T2 = None
    trip_id1 = t1['trip_id']
    trip_id2 = t2['trip_id']
    trip_time1 = float(t1.get('trip_time', 0.0) or 0.0)
    trip_time2 = float(t2.get('trip_time', 0.0) or 0.0)
    start_time1 = t1['start_time']
    start_time2 = t2['start_time']

    # 计算最大容忍等待时间（与 ST_3DIntersects 阶段一致，严格约束）
    max_wait_time1 = min(WAIT_RATIO * trip_time1, WAIT_CAP_SECS)
    max_wait_time2 = min(WAIT_RATIO * trip_time2, WAIT_CAP_SECS)

    tt_count = 0  # 成功生成的TT文件数量
    skipped_count = 0  # 跳过的已存在文件数量

    try:
        # 1. 从缓存加载两个prism
        mesh_T1 = prism_cache.checkout_mesh(trip_id1)
        if mesh_T1 is None:
            prism_path1 = prism_cache.checkout(trip_id1)
            if not prism_path1:
                return (
                    i, j, 0, time.perf_counter() - pair_start_time, False, 'prism_cache',
                    f"行程 {trip_id1} 棱柱缓存失效")
            prism_cache.release(trip_id1)
            return (
                i, j, 0, time.perf_counter() - pair_start_time, False, 'prism_load',
                f"行程 {trip_id1} 棱柱加载失败（文件存在但无法加载）")
        else:
            prism_cache.checkout(trip_id1)

        mesh_T2 = prism_cache.checkout_mesh(trip_id2)
        if mesh_T2 is None:
            prism_path2 = prism_cache.checkout(trip_id2)
            if not prism_path2:
                prism_cache.release(trip_id1)
                return (
                    i, j, 0, time.perf_counter() - pair_start_time, False, 'prism_cache',
                    f"行程 {trip_id2} 棱柱缓存失效")
            prism_cache.release(trip_id1)
            prism_cache.release(trip_id2)
            return (
                i, j, 0, time.perf_counter() - pair_start_time, False, 'prism_load',
                f"行程 {trip_id2} 棱柱加载失败（文件存在但无法加载）")
        else:
            prism_cache.checkout(trip_id2)

        if mesh_T1 is None or mesh_T2 is None:
            prism_cache.release(trip_id1)
            prism_cache.release(trip_id2)
            return (i, j, 0, time.perf_counter() - pair_start_time, False, 'prism_load', "prism加载后为None")

        # 2. 计算基础时间偏移
        delta_t1 = start_time1 - base_time
        delta_t2 = start_time2 - base_time

        # 创建mesh的副本用于平移（避免修改原始mesh）
        mesh_T1_base = mesh_T1.copy()
        mesh_T2_base = mesh_T2.copy()
        if delta_t1 != 0:
            mesh_T1_base.apply_translation([0, 0, delta_t1])
        if delta_t2 != 0:
            mesh_T2_base.apply_translation([0, 0, delta_t2])

        conn = None
        try:
            conn = _get_worker_connection()

            # 3. 处理第一种TT：先接trip1，再接trip2（O1→O2）
            # 如果 ST_3DIntersects 阶段已经判定该顺序不可能有交集，则直接跳过，减少pymeshlab调用
            if use_o1o2 and o1_to_o2_time is not None:
                mesh_T2_translated = None
                vehicle_arrival_at_o2 = start_time1 + o1_to_o2_time
                diff_o1o2 = vehicle_arrival_at_o2 - start_time2  # >0 表示乘客2在等，<0 表示车辆（带乘客1）在等
                wait_time_o1o2 = abs(diff_o1o2)

                # 在各自最大等待上限内截断后再进行平移
                if diff_o1o2 == 0:
                    wait_offset = 0.0
                else:
                    if diff_o1o2 < 0:
                        # 车辆（带乘客1）在 O2 等待乘客2，受乘客1等待上限约束
                        capped_wait = min(wait_time_o1o2, max_wait_time1)
                    else:
                        # 乘客2在 O2 等车，受乘客2等待上限约束
                        capped_wait = min(wait_time_o1o2, max_wait_time2)
                    wait_offset = math.copysign(capped_wait, diff_o1o2)

                mesh_T2_translated = mesh_T2_base.copy()
                if wait_offset != 0:
                    mesh_T2_translated.apply_translation([0, 0, wait_offset])

                if mesh_T2_translated is not None:
                    os.makedirs(tt_dir, exist_ok=True)
                    tt_filename = f"tt_{trip_id1}_{trip_id2}_O1O2.ply"
                    tt_path = os.path.join(tt_dir, tt_filename)
                    metadata_filename = f"tt_{trip_id1}_{trip_id2}_O1O2.json"
                    metadata_path = os.path.join(tt_dir, metadata_filename)

                    # 更稳健的“跳过已存在”：
                    # - ply+json 都在：直接跳过
                    # - 仅 ply 在：补写 json（不再重算交集）
                    # - 仅 json 在：重算 ply（并覆盖/更新 json）
                    tt_exists = os.path.exists(tt_path)
                    meta_exists = os.path.exists(metadata_path)
                    if tt_exists and meta_exists:
                        tt_count += 1
                        skipped_count += 1
                    else:
                        metadata = {
                            "trip_id1": trip_id1,
                            "trip_id2": trip_id2,
                            "direction": "O1O2",
                            "wait_offset": wait_offset,
                            "wait_time": wait_time_o1o2,
                            "max_wait_time1": max_wait_time1,
                            "max_wait_time2": max_wait_time2,
                            "wait_ratio": WAIT_RATIO,
                            "wait_cap_secs": WAIT_CAP_SECS,
                            "vehicle_arrival_at_o2": vehicle_arrival_at_o2,
                            "start_time2": start_time2,
                        }
                        if tt_exists and (not meta_exists):
                            # 只缺元数据：补写元数据即可
                            try:
                                with open(metadata_path, 'w', encoding='utf-8') as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                                tt_count += 1
                                skipped_count += 1
                            except Exception:
                                pass
                        else:
                            # 缺 ply（或两者都缺）：重算交集并落盘
                            TT = compute_mesh_intersection(mesh_T1_base, mesh_T2_translated)
                            if TT is not None and hasattr(TT, 'vertices') and hasattr(TT, 'faces') and len(
                                    TT.vertices) > 0 and len(TT.faces) > 0:
                                try:
                                    TT.export(tt_path, file_type="ply")
                                    with open(metadata_path, 'w', encoding='utf-8') as f:
                                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                                    tt_count += 1
                                except Exception:
                                    pass  # 保存失败，继续处理下一种TT

            # 4. 处理第二种TT：先接trip2，再接trip1（O2→O1）
            # 同理，只有在 ST_3DIntersects 阶段认为该顺序可能有交集时才计算
            if use_o2o1 and o2_to_o1_time is not None:
                mesh_T1_translated = None
                vehicle_arrival_at_o1 = start_time2 + o2_to_o1_time
                diff_o2o1 = vehicle_arrival_at_o1 - start_time1  # >0 表示乘客1在等，<0 表示车辆（带乘客2）在等
                wait_time_o2o1 = abs(diff_o2o1)

                # 同样在允许的最大等待范围内进行截断平移
                if diff_o2o1 == 0:
                    wait_offset = 0.0
                else:
                    if diff_o2o1 < 0:
                        # 车辆（带乘客2）在 O1 等待乘客1，受乘客2等待上限约束
                        capped_wait = min(wait_time_o2o1, max_wait_time2)
                    else:
                        # 乘客1在 O1 等车，受乘客1等待上限约束
                        capped_wait = min(wait_time_o2o1, max_wait_time1)
                    wait_offset = math.copysign(capped_wait, diff_o2o1)

                mesh_T1_translated = mesh_T1_base.copy()
                if wait_offset != 0:
                    mesh_T1_translated.apply_translation([0, 0, wait_offset])

                if mesh_T1_translated is not None:
                    os.makedirs(tt_dir, exist_ok=True)
                    tt_filename = f"tt_{trip_id1}_{trip_id2}_O2O1.ply"
                    tt_path = os.path.join(tt_dir, tt_filename)
                    metadata_filename = f"tt_{trip_id1}_{trip_id2}_O2O1.json"
                    metadata_path = os.path.join(tt_dir, metadata_filename)

                    tt_exists = os.path.exists(tt_path)
                    meta_exists = os.path.exists(metadata_path)
                    if tt_exists and meta_exists:
                        tt_count += 1
                        skipped_count += 1
                    else:
                        metadata = {
                            "trip_id1": trip_id1,
                            "trip_id2": trip_id2,
                            "direction": "O2O1",
                            "wait_offset": wait_offset,
                            "wait_time": wait_time_o2o1,
                            "max_wait_time1": max_wait_time1,
                            "max_wait_time2": max_wait_time2,
                            "wait_ratio": WAIT_RATIO,
                            "wait_cap_secs": WAIT_CAP_SECS,
                            "vehicle_arrival_at_o1": vehicle_arrival_at_o1,
                            "start_time1": start_time1,
                        }
                        if tt_exists and (not meta_exists):
                            try:
                                with open(metadata_path, 'w', encoding='utf-8') as f:
                                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                                tt_count += 1
                                skipped_count += 1
                            except Exception:
                                pass
                        else:
                            TT = compute_mesh_intersection(mesh_T1_translated, mesh_T2_base)
                            if TT is not None and hasattr(TT, 'vertices') and hasattr(TT, 'faces') and len(
                                    TT.vertices) > 0 and len(TT.faces) > 0:
                                try:
                                    TT.export(tt_path, file_type="ply")
                                    with open(metadata_path, 'w', encoding='utf-8') as f:
                                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                                    tt_count += 1
                                except Exception:
                                    pass  # 保存失败，继续处理

        finally:
            if conn is not None:
                _return_worker_connection(conn)

        # 5. 释放缓存引用并返回
        prism_cache.release(trip_id1)
        prism_cache.release(trip_id2)

        if tt_count > 0:
            return (i, j, tt_count, time.perf_counter() - pair_start_time, True, None, None, skipped_count)
        else:
            return (i, j, 0, time.perf_counter() - pair_start_time, False, 'tt', "两种TT顺序都未通过过滤或无交集",
                    skipped_count)

    except Exception as e:
        try:
            prism_cache.release(trip_id1)
            prism_cache.release(trip_id2)
        except:
            pass
        return (i, j, 0, time.perf_counter() - pair_start_time, False, 'unknown', str(e), 0)


def main(
    checkpoint_name: Optional[str] = None,
    resume: bool = True,
    csv_path: str = r'D:\shirou\carpool\data\1375704000_30min.csv',
    start_ts: int = 1375704000,
    slice_len: int = 9645,
    max_trips: int = 12000,
) -> None:
    """
    批量处理主函数（使用进程池优化）

    Args:
        checkpoint_name: 检查点名称，如果为None则使用默认名称（基于CSV文件名）
        resume: 是否从检查点恢复，如果为False则从头开始
        csv_path: 输入CSV路径
        start_ts: 起始时间戳（秒）
        slice_len: 从 start_ts 命中的第一条开始截取的行数
        max_trips: 最终用于处理的最大行程数（在 slice 后再截断）
    """
    # 开始性能监控
    monitor = start_monitoring()

    try:
        tt_started = False  # 标记是否已经进入TT阶段（用于异常时保存TT检查点）
        start_time0 = time.time()

        # 初始化检查点管理器
        if checkpoint_name is None:
            # 使用CSV文件名作为默认检查点名称
            csv_path = r'D:\shirou\carpool\data\1375704000_30min.csv'
            csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
            checkpoint_name = f"checkpoint_{csv_basename}"

        checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR, checkpoint_name)

        # 尝试加载检查点
        checkpoint_data = None
        if resume:
            checkpoint_data = checkpoint_mgr.load()
            if checkpoint_data:
                print(f"[检查点] 发现检查点，将从阶段 '{checkpoint_data['stage']}' 恢复")
            else:
                print("[检查点] 未发现检查点，从头开始处理")

        # 1. 批量读取CSV（建议通过 CLI 覆盖这些参数，便于论文复现）
        monitor.record_stage_start("数据读取")
        start_time_read = time.time()
        # csv_path = r'D:\shirou\carpool\data\setpoins.csv'
        # df = pd.read_csv(csv_path)
        # print(f"读取CSV文件，共 {len(df)} 行")
        # csv_path 由 main 参数传入
        # csv_path = r'D:\shirou\carpool\data\1375740000_30min.csv'
        df = pd.read_csv(csv_path)
        # start_ts = 1375354800
        # start_ts = 1375358400
        # start_ts 由 main 参数传入
        df = df.sort_values('pickup_datetime').reset_index(drop=True)
        valid_idx = df.index[df['pickup_datetime'] >= start_ts]
        if len(valid_idx) == 0:
            df = df.iloc[0:0].copy()
        else:
            start_pos = int(valid_idx[0])
            # end_pos = start_pos + 11813
            end_pos = start_pos + int(slice_len)
            df = df.iloc[start_pos:end_pos].reset_index(drop=True)
        print(f"共 {len(df)} 行数据")

        # 统一生成秒级时间戳
        last_trip_ts = None
        if len(df) > 0:
            if np.issubdtype(df['pickup_datetime'].dtype, np.number):
                df = df.assign(pickup_timestamp=df['pickup_datetime'].astype('int64'))
            else:
                ts = pd.to_datetime(df['pickup_datetime'], errors='coerce')
                ts_sec = (ts.astype('int64') // 10 ** 9)
                df = df.assign(pickup_timestamp=ts_sec)
            df = df.dropna(subset=['pickup_timestamp']).copy()
            df['pickup_timestamp'] = df['pickup_timestamp'].astype(np.int64)
            if len(df) > 0:
                last_trip_ts = int(df['pickup_timestamp'].iloc[-1])
        if last_trip_ts is not None:
            print(f"最后一个行程的时间戳: {last_trip_ts}")
        elif len(df) == 0:
            print("未找到满足条件的行程，无法提供最后一个时间戳")

        df = df.head(int(max_trips)).reset_index(drop=True)
        end_time_read = time.time()
        monitor.record_stage_end("数据读取")
        print(f"读取数据用时: {end_time_read - start_time_read:.2f}秒")

        # 2. 预处理所有行程参数（批量OD点处理优化）
        monitor.record_stage_start("数据预处理")
        start_time_process = time.time()
        pickup_points = df[['pickup_longitude', 'pickup_latitude']].values
        dropoff_points = df[['dropoff_longitude', 'dropoff_latitude']].values
        all_points = np.vstack([pickup_points, dropoff_points])
        all_points_list = [tuple(x) for x in all_points]

        # 批量查找所有OD点的最近节点
        conn = get_main_db_connection()
        node_results = get_multiple_nearest_nodes_with_coords_batch(conn, all_points_list)
        return_main_db_connection(conn)

        n = len(df)
        pickup_nodes = node_results[:n]
        dropoff_nodes = node_results[n:]

        trip_list = []
        # 额外行驶时间（detour）约束：用于构造棱柱时间上界 max_trip_time
        max_extra_secs = float(DETOUR_CAP_SECS)
        for idx, row in df.iterrows():
            O = (row['pickup_longitude'], row['pickup_latitude'])
            D = (row['dropoff_longitude'], row['dropoff_latitude'])
            trip_time = float(row['trip_time_in_secs'])
            start_time = int(row['pickup_timestamp'])
            O_node_id, O_node = pickup_nodes[idx]
            D_node_id, D_node = dropoff_nodes[idx]

            # 计算包含容忍度的最大行程时间（论文式：ratio + cap）
            extra_time = min(max_extra_secs, float(DETOUR_RATIO) * trip_time)
            max_trip_time = trip_time + extra_time

            trip_list.append({
                'trip_id': idx,
                'O': O,
                'D': D,
                'trip_time': trip_time,  # 原始行程时间（真实值）
                'max_trip_time': max_trip_time,  # 包含容忍度的最大时间
                'start_time': start_time,
                'O_node_id': O_node_id,
                'O_node': O_node,
                'D_node_id': D_node_id,
                'D_node': D_node
            })
        end_time_process = time.time()
        monitor.record_stage_end("数据预处理")
        print(f"数据预处理用时: {end_time_process - start_time_process:.2f}秒")

        # 在进入 ST_3DIntersects 之前，一次性扫描一遍所有有效 prism 文件的 trip_id 集合，
        # 把不在集合里的 trip 直接过滤掉，不让它们进入 worker。
        # 注意：这里只负责过滤“候选对”层面，trip_list 仍然保留，以兼容已有逻辑。
        valid_prism_trip_ids = get_available_prism_trip_ids()

        # 检查点 + 候选对与行驶时间预计算
        # 所有行程对 (i < j) 直接进入 ST_3DIntersects + 等待时间逻辑。
        # 为了与当前两阶段结构对齐，这里将“进入 ST_3DIntersects 之前的所有候选对”统一称为 candidate_pairs。

        # 所有进入 ST_3DIntersects 的候选对列表
        candidate_pairs: List[Tuple[int, int]] = []
        # 预计算的 O1→O2 / O2→O1 行驶时间映射（(i, j) -> dict）
        pair_travel_times: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}

        if checkpoint_data and checkpoint_data['stage'] in ['candidate', 'st3d', 'tt']:
            # 从检查点恢复候选对和已预计算好的行驶时间
            candidate_pairs = checkpoint_data['data'].get('candidate_pairs', [])
            pair_travel_times = checkpoint_data['data'].get('pair_travel_times', {}) or {}
            print(
                f"[检查点] 从检查点恢复 candidate_pairs: {len(candidate_pairs)} 对，跳过候选生成与行驶时间预计算")

            # 即使从旧检查点恢复，这里也要套一层“棱柱存在性预筛选”，
            # 否则旧检查点里仍然含有大量没有棱柱文件的行程对。
            if candidate_pairs and valid_prism_trip_ids:
                before_cnt = len(candidate_pairs)

                # 过滤掉两端没有棱柱文件的行程对
                filtered_pairs = [
                    (i, j) for (i, j) in candidate_pairs
                    if (i in valid_prism_trip_ids) and (j in valid_prism_trip_ids)
                ]
                candidate_pairs = filtered_pairs

                # 同步裁剪行驶时间映射：按 trip_id 集合再过滤一遍，避免构造巨大 set(filtered_pairs)
                if pair_travel_times:
                    pair_travel_times = {
                        (i, j): t
                        for (i, j), t in pair_travel_times.items()
                        if (i in valid_prism_trip_ids) and (j in valid_prism_trip_ids)
                    }

                print(
                    f"[预筛选] 从检查点恢复的候选对中，仅保留两端已有棱柱的行程对: "
                    f"{len(candidate_pairs)}/{before_cnt} 对")
        else:
            # 直接生成所有行程对 (i < j) 作为 ST_3DIntersects 的候选，
            # 但要求两个行程的 trip_id 都已经有对应的棱柱文件（prism），
            # 否则不让它们进入 ST_3DIntersects worker，避免无效 I/O。
            print("生成所有行程对作为候选（仅保留两端都有棱柱文件的行程对）...")
            monitor.record_stage_start("候选对生成（全量）")
            all_pairs_start_time = time.time()

            candidate_pairs = []
            for i in range(n):
                if i not in valid_prism_trip_ids:
                    continue
                for j in range(i + 1, n):
                    if j not in valid_prism_trip_ids:
                        continue
                    candidate_pairs.append((i, j))

            all_pairs_end_time = time.time()
            monitor.record_stage_end("候选对生成（全量）")
            print(
                f"候选对总数（两端都有棱柱文件的 i<j）：{len(candidate_pairs)} 对，生成耗时: {all_pairs_end_time - all_pairs_start_time:.2f} 秒")

            # ===== 在进入 ST_3DIntersects / TT 之前，预计算所有 candidate_pairs 的 O1→O2 和 O2→O1 行驶时间 =====
            print("[预计算] 开始为所有候选行程对批量预计算 O1→O2 和 O2→O1 的最短驾驶时间...")
            travel_time_precompute_start = time.time()
            pair_travel_times = {}
            if candidate_pairs:
                conn = get_main_db_connection()
                try:
                    # 用 (i, j) 直接绑定到对应的节点对，避免依赖 idx/列表位置对齐导致错配
                    # 同时对节点对去重，减少重复最短路计算
                    o1o2_pairs_by_ij: Dict[Tuple[int, int], Tuple[int, int]] = {}
                    o2o1_pairs_by_ij: Dict[Tuple[int, int], Tuple[int, int]] = {}

                    # 去重集合
                    unique_o1o2_pairs = set()
                    unique_o2o1_pairs = set()

                    for (i, j) in candidate_pairs:
                        t1 = trip_list[i]
                        t2 = trip_list[j]
                        o1_node_id = t1.get('O_node_id')
                        o2_node_id = t2.get('O_node_id')
                        if o1_node_id is not None and o2_node_id is not None:
                            key = (i, j)
                            pair_o1o2 = (o1_node_id, o2_node_id)
                            pair_o2o1 = (o2_node_id, o1_node_id)
                            o1o2_pairs_by_ij[key] = pair_o1o2
                            o2o1_pairs_by_ij[key] = pair_o2o1
                            unique_o1o2_pairs.add(pair_o1o2)
                            unique_o2o1_pairs.add(pair_o2o1)

                    travel_times_o1o2: Dict[Tuple[int, int], Optional[float]] = {}
                    travel_times_o2o1: Dict[Tuple[int, int], Optional[float]] = {}
                    # 分块批量查询，避免 end_vids 过大造成 SQL/内存压力（可按实际情况调大/调小）
                    chunk_size = 5000000

                    unique_o1o2_list = list(unique_o1o2_pairs)
                    unique_o2o1_list = list(unique_o2o1_pairs)

                    if unique_o1o2_list:
                        print(f"[预计算] 批量查询 {len(unique_o1o2_list)} 个(去重后) O1→O2 节点对的驾驶时间...")
                        for start in range(0, len(unique_o1o2_list), chunk_size):
                            chunk = unique_o1o2_list[start:start + chunk_size]
                            travel_times_o1o2.update(batch_get_shortest_travel_times(conn, chunk))
                    if unique_o2o1_list:
                        print(f"[预计算] 批量查询 {len(unique_o2o1_list)} 个(去重后) O2→O1 节点对的驾驶时间...")
                        for start in range(0, len(unique_o2o1_list), chunk_size):
                            chunk = unique_o2o1_list[start:start + chunk_size]
                            travel_times_o2o1.update(batch_get_shortest_travel_times(conn, chunk))

                    # 回填到每个 (i, j)，若缺失则保持 None
                    for (i, j) in candidate_pairs:
                        key = (i, j)
                        o1o2_pair = o1o2_pairs_by_ij.get(key)
                        o2o1_pair = o2o1_pairs_by_ij.get(key)
                        pair_travel_times[key] = {
                            'o1_to_o2_time': travel_times_o1o2.get(o1o2_pair) if o1o2_pair else None,
                            'o2_to_o1_time': travel_times_o2o1.get(o2o1_pair) if o2o1_pair else None,
                        }
                finally:
                    return_main_db_connection(conn)

            travel_time_precompute_time = time.time() - travel_time_precompute_start
            print(f"[预计算] 行驶时间预计算完成，耗时 {travel_time_precompute_time:.2f} 秒")

            # 保存候选阶段完成检查点（包含所有候选对和预计算的行驶时间）
            checkpoint_mgr.save('candidate', {
                'candidate_pairs': candidate_pairs,
                'pair_travel_times': pair_travel_times
            }, {
                                    'total_pairs': len(candidate_pairs),
                                    'csv_path': csv_path,
                                    'start_ts': start_ts,
                                    'slice_len': int(slice_len),
                                    'max_trips': int(max_trips),
                                    'detour_ratio': DETOUR_RATIO,
                                    'detour_cap_secs': DETOUR_CAP_SECS,
                                    'wait_ratio': WAIT_RATIO,
                                    'wait_cap_secs': WAIT_CAP_SECS,
                                })

        if not candidate_pairs:
            print("没有可用于 ST_3DIntersects 的候选行程对，跳过该阶段。")
            return

        # ========== ST_3DIntersects 阶段：快速几何过滤 ==========
        processed_tt_count = 0  # 已处理的TT任务数量（用于检查点恢复）
        if checkpoint_data and checkpoint_data['stage'] == 'tt':
            # 从TT阶段检查点恢复时，直接使用已有的 st3d 通过对和预计算结果，跳过 ST_3DIntersects 计算
            if 'candidate_pairs' in checkpoint_data['data']:
                candidate_pairs = checkpoint_data['data'].get('candidate_pairs', [])
            # 兼容旧检查点字段：level4_pass_pairs
            st3d_pass_pairs = checkpoint_data['data'].get(
                'st3d_pass_pairs',
                checkpoint_data['data'].get('level4_pass_pairs', [])
            )
            # 恢复预计算的行驶时间和 ST_3DIntersects 顺序通过信息
            pair_travel_times = checkpoint_data['data'].get('pair_travel_times', {})
            st3d_pass_orders = checkpoint_data['data'].get('st3d_pass_orders', {})
            st3d_filtered = checkpoint_data['metadata'].get(
                'st3d_filtered',
                checkpoint_data['metadata'].get('level4_filtered', 0)
            )
            processed_tt_count = checkpoint_data['metadata'].get('processed_tt_count', 0)
            print(
                f"[检查点] 从TT阶段检查点恢复: st3d_pass_pairs={len(st3d_pass_pairs)} 对，已处理TT任务: {processed_tt_count}/{len(st3d_pass_pairs)}，将直接进入TT计算（跳过ST_3DIntersects）")
        else:
            print("开始 ST_3DIntersects 阶段：快速几何过滤...")
            monitor.record_stage_start("ST_3DIntersects筛选")
            fourth_stage_start_time = time.time()

            # 检查是否有 ST_3DIntersects 阶段的检查点（从候选阶段检查点恢复）
            st3d_filtered = 0
            st3d_pass_pairs = []
            processed_st3d_count = 0
            # 用于记录每个行程对在两种顺序下是否通过 ST_3DIntersects 过滤
            st3d_pass_orders = {}
            # 更细粒度统计：解释 st3d_filtered 的来源
            st3d_stats = {
                # 过滤原因（对级别）
                'filtered_total': 0,
                'filtered_no_attempt_both': 0,  # 两个顺序都未实际执行 ST_3DIntersects（缺时/等待超限导致）
                'filtered_tested_no_intersect': 0,  # 至少一个顺序执行了 ST_3DIntersects，但两个顺序最终都不相交
                # 原因细分（按“顺序”累计，o1o2 与 o2o1 各自计数都会进来）
                'order_missing_travel_time': 0,
                'order_wait_exceed': 0,
                'order_tested_no_intersect': 0,
                'order_missing_mesh': 0,
                'order_mesh_load_failed': 0,
                'order_pass': 0,
                'order_error': 0,
                # 通过情况（对级别）
                'pair_pass_any': 0,
                'pair_pass_both': 0,
                'pair_error_fallback_pass': 0,
            }
            # 统计通过时间轴+几何预筛选，真正进入 ST_3DIntersects 计算的行程对数量
            prefilter_pass_pairs = 0
            if checkpoint_data and checkpoint_data['stage'] == 'st3d':
                # 从 ST_3DIntersects 检查点恢复时，也需要恢复 candidate_pairs 和预计算结果
                if 'candidate_pairs' in checkpoint_data['data']:
                    candidate_pairs = checkpoint_data['data'].get('candidate_pairs', [])
                if 'pair_travel_times' in checkpoint_data['data']:
                    pair_travel_times = checkpoint_data['data'].get('pair_travel_times', {})
                st3d_pass_pairs = checkpoint_data['data'].get(
                    'st3d_pass_pairs',
                    checkpoint_data['data'].get('level4_pass_pairs', [])
                )
                # 恢复已记录的顺序通过信息
                st3d_pass_orders = checkpoint_data['data'].get('st3d_pass_orders', {})
                processed_st3d_count = checkpoint_data['metadata'].get('processed_st3d_count', 0)
                st3d_filtered = checkpoint_data['metadata'].get(
                    'st3d_filtered',
                    checkpoint_data['metadata'].get('level4_filtered', 0)
                )
                print(
                    f"[检查点] 从检查点恢复 ST_3DIntersects 通过的行程对: {len(st3d_pass_pairs)} 对，已处理: {processed_st3d_count}/{len(candidate_pairs)}")

            # 如果从检查点恢复，需要跳过已处理的对
            remaining_candidate_pairs = candidate_pairs[
                processed_st3d_count:] if processed_st3d_count > 0 else candidate_pairs

            def iter_st3d_tasks():
                nonlocal prefilter_pass_pairs
                for (i, j) in remaining_candidate_pairs:
                    t1 = trip_list[i]
                    t2 = trip_list[j]

                    # ---------- 预筛选 1：时间轴区间快速检测 ----------
                    # 使用 [start_time, start_time + max_trip_time + max_wait_time] 的区间重叠性
                    # 若两行程的活动时间区间完全不重叠，则在等待约束下几乎不可能产生可行拼车，直接剪枝
                    if not _time_intervals_overlap_with_wait(t1, t2):
                        continue

                    # ---------- 预筛选 2：几何距离过滤 ----------
                    # 两个方向都判断：
                    # - O2 到行程1路径（O1→D1）的“时间距离”是否 <= 行程1的 max_trip_time
                    # - O1 到行程2路径（O2→D2）的“时间距离”是否 <= 行程2的 max_trip_time
                    # 只要满足任意一个方向的条件，就通过几何过滤
                    if not (_o2_near_o1d1_segment(t1, t2) or _o1_near_o2d2_segment(t1, t2)):
                        continue

                    # 通过所有预筛选，计数 +1
                    prefilter_pass_pairs += 1

                    base_time = min(trip_list[i]['start_time'], trip_list[j]['start_time'])
                    travel_times = pair_travel_times.get((i, j), {})
                    yield (
                        i,
                        j,
                        base_time,
                        travel_times.get('o1_to_o2_time'),
                        travel_times.get('o2_to_o1_time'),
                    )

            with ProcessPoolExecutor(max_workers=EVALUATION_WORKERS, initializer=_worker_initializer) as pool:
                inflight = set()
                task_iter = iter_st3d_tasks()

                def submit_st3d_next():
                    try:
                        i, j, base_time, o1_to_o2_time, o2_to_o1_time = next(task_iter)
                    except StopIteration:
                        return False
                    payload = {
                        'i': i,
                        'j': j,
                        't1': trip_list[i],
                        't2': trip_list[j],
                        'base_time': base_time,
                        'o1_to_o2_time': o1_to_o2_time,
                        'o2_to_o1_time': o2_to_o1_time,
                    }
                    fut = pool.submit(_st3d_filter_worker, payload)
                    inflight.add(fut)
                    return True

                # 预填充任务
                while len(inflight) < min(max_inflight, len(candidate_pairs)) and submit_st3d_next():
                    pass
                print(f"[ST_3DIntersects] 初始化提交 {len(inflight)} 个任务（最大并发 {max_inflight}），开始快速过滤...")

                progress_total = len(candidate_pairs)
                progress_step = 100000
                idx = 0
                last_progress_time = fourth_stage_start_time  # 记录上次进度输出的时间

                while inflight:
                    done = next(as_completed(inflight))
                    inflight.remove(done)
                    idx += 1
                    try:
                        i, j, has_intersect, elapsed_time, pass_o1o2, pass_o2o1, attempted_o1o2, attempted_o2o1, reason_o1o2, reason_o2o1 = done.result()
                        # 记录两种顺序下的通过情况
                        st3d_pass_orders[(i, j)] = {
                            'o1o2': bool(pass_o1o2),
                            'o2o1': bool(pass_o2o1),
                        }

                        # 细粒度统计（顺序级别）
                        for r in (reason_o1o2, reason_o2o1):
                            if r == "missing_travel_time":
                                st3d_stats['order_missing_travel_time'] += 1
                            elif r == "wait_exceed":
                                st3d_stats['order_wait_exceed'] += 1
                            elif r == "no_intersect":
                                st3d_stats['order_tested_no_intersect'] += 1
                            elif r == "missing_mesh":
                                st3d_stats.setdefault('order_missing_mesh', 0)
                                st3d_stats['order_missing_mesh'] += 1
                            elif r == "mesh_load_failed":
                                st3d_stats.setdefault('order_mesh_load_failed', 0)
                                st3d_stats['order_mesh_load_failed'] += 1
                            elif r == "pass":
                                st3d_stats['order_pass'] += 1
                            elif r == "error":
                                st3d_stats['order_error'] += 1

                        # 细粒度统计（对级别）
                        if bool(pass_o1o2) and bool(pass_o2o1):
                            st3d_stats['pair_pass_both'] += 1
                        if bool(pass_o1o2) or bool(pass_o2o1):
                            st3d_stats['pair_pass_any'] += 1
                        if reason_o1o2 == "error" or reason_o2o1 == "error":
                            # worker 内部异常保守放行（或者外层兜底返回 error）
                            st3d_stats['pair_error_fallback_pass'] += 1

                        if has_intersect:
                            st3d_pass_pairs.append((i, j))
                        else:
                            st3d_filtered += 1
                            st3d_stats['filtered_total'] += 1
                            if (not attempted_o1o2) and (not attempted_o2o1):
                                st3d_stats['filtered_no_attempt_both'] += 1
                            else:
                                st3d_stats['filtered_tested_no_intersect'] += 1

                        # 定期保存检查点（每处理1000000个任务自动保存）
                        current_processed = processed_st3d_count + idx
                        if current_processed % 1000000 == 0:
                            checkpoint_mgr.save(
                                'st3d',
                                {
                                    'st3d_pass_pairs': st3d_pass_pairs,
                                    'pair_travel_times': pair_travel_times,
                                    'st3d_pass_orders': st3d_pass_orders,
                                },
                                {
                                    'processed_st3d_count': current_processed,
                                    'st3d_filtered': st3d_filtered,
                                    'total': len(candidate_pairs),
                                },
                            )

                        if idx % progress_step == 0 or idx == progress_total:
                            current_time = time.time()
                            total_elapsed = current_time - fourth_stage_start_time
                            recent_elapsed = current_time - last_progress_time
                            if idx % progress_step == 0:
                                # 正好完成100000个的倍数
                                recent_count = progress_step
                            else:
                                # 最后一批不足100000个，按比例换算到100000个
                                recent_count = idx % progress_step
                            print(
                                f"[ST_3DIntersects] 进度: {current_processed}/{len(candidate_pairs)}，通过: {len(st3d_pass_pairs)}，过滤: {st3d_filtered}，"
                                f"总耗时: {total_elapsed:.2f}秒，最近{recent_count}个耗时: {recent_elapsed:.2f}秒\n"
                                f"    [过滤拆分] tested_no_intersect={st3d_stats['filtered_tested_no_intersect']}，"
                                f"no_attempt_both={st3d_stats['filtered_no_attempt_both']}（missing_time={st3d_stats['order_missing_travel_time']}，wait_exceed={st3d_stats['order_wait_exceed']}，missing_mesh={st3d_stats['order_missing_mesh']}，mesh_load_failed={st3d_stats['order_mesh_load_failed']}）\n"
                                f"    [顺序通过] pass_any={st3d_stats['pair_pass_any']}，pass_both={st3d_stats['pair_pass_both']}，error_fallback_pass={st3d_stats['pair_error_fallback_pass']}")
                            last_progress_time = current_time
                    except Exception as e:
                        # 异常时保守放行
                        i, j = None, None
                        try:
                            i, j, *_ = done.result()
                        except:
                            pass
                        if i is not None and j is not None:
                            st3d_pass_pairs.append((i, j))
                            st3d_stats['pair_error_fallback_pass'] += 1
                        print(f"[ST_3DIntersects] 异常: {e}")

                    # 补充新的任务
                    while len(inflight) < max_inflight and submit_st3d_next():
                        pass

            fourth_stage_end_time = time.time()
            monitor.record_stage_end("ST_3DIntersects筛选")
            print(f"ST_3DIntersects 阶段筛掉: {st3d_filtered} 对")
            print(f"ST_3DIntersects 阶段通过: {len(st3d_pass_pairs)} 对（进入TT计算）")
            print(f"时间轴+几何预筛选阶段通过的行程对: {prefilter_pass_pairs} 对（进入 ST_3DIntersects 计算）")
            print(f"ST_3DIntersects 阶段筛选耗时: {fourth_stage_end_time - fourth_stage_start_time:.2f} 秒")

            # 保存 ST_3DIntersects 阶段完成检查点
            checkpoint_mgr.save(
                'st3d',
                {
                    'st3d_pass_pairs': st3d_pass_pairs,
                    'pair_travel_times': pair_travel_times,
                    'st3d_pass_orders': st3d_pass_orders,
                },
                {
                    'processed_st3d_count': len(candidate_pairs),
                    'st3d_filtered': st3d_filtered,
                    'total': len(candidate_pairs),
                },
            )

        if not st3d_pass_pairs:
            print("没有通过 ST_3DIntersects 筛选的行程对，跳过TT计算。")
            return

        # ========== TT计算（pymeshlab交集计算，考虑等待时间） ==========
        # 在进入TT阶段前保存TT检查点（包含所有 ST_3DIntersects 通过对 和预计算结果），用于后续直接跳过前面各阶段
        # 如果从检查点恢复，processed_tt_count已经在上面恢复
        if processed_tt_count == 0:
            checkpoint_mgr.save(
                'tt',
                {
                    'candidate_pairs': candidate_pairs,
                    'st3d_pass_pairs': st3d_pass_pairs,
                    'pair_travel_times': pair_travel_times,
                    'st3d_pass_orders': st3d_pass_orders,
                },
                {
                    'total_st3d_pairs': len(st3d_pass_pairs),
                    'processed_tt_count': 0,
                },
            )
        tt_started = True
        monitor.record_stage_start("TT计算（考虑等待时间）")
        os.makedirs(TT_OUTPUT_DIR, exist_ok=True)
        print(f"开始TT计算（考虑等待时间的两种TT生成），共 {len(st3d_pass_pairs)} 对行程，输出目录：{TT_OUTPUT_DIR}")
        if processed_tt_count > 0:
            print(f"[TT] 从检查点恢复，已处理 {processed_tt_count} 个任务，将从索引 {processed_tt_count} 开始继续处理")

        # 批量预计算O1→O2和O2→O1的最短驾驶时间
        travel_time_precompute_start = time.time()

        # 这里的 pair_travel_times 已经在候选阶段预计算完成，并在 candidate/st3d/tt 检查点中被恢复。
        # 为了避免重复的批量最短路查询，这里不再访问数据库，而是仅在原有的
        # pair_travel_times 上，按当前 TT 还需要处理的 st3d_pass_pairs[processed_tt_count:]
        # 构造一个子集映射，供后续 TT 计算使用。
        filtered_pair_travel_times = {}
        missing_count = 0
        for (i, j) in st3d_pass_pairs[processed_tt_count:]:
            tt_key = (i, j)
            tt_times = pair_travel_times.get(tt_key)
            if tt_times is not None:
                filtered_pair_travel_times[tt_key] = tt_times
            else:
                # 理论上第三层预计算时只对有 O_node_id 的行程对做了查询，
                # 这里如果缺失，说明第三层阶段就没有对应的最短路结果，保持 None 即可。
                missing_count += 1

        pair_travel_times = filtered_pair_travel_times
        travel_time_precompute_time = time.time() - travel_time_precompute_start
        print(
            f"[TT] 行驶时间映射复用完成，共 {len(pair_travel_times)} 对，有 {missing_count} 对缺少预计算结果，耗时 {travel_time_precompute_time:.2f} 秒")

        def iter_tt_tasks():
            # 从processed_tt_count位置开始迭代
            for (i, j) in st3d_pass_pairs[processed_tt_count:]:
                base_time = min(trip_list[i]['start_time'], trip_list[j]['start_time'])
                travel_times = pair_travel_times.get((i, j), {})
                order_flags = st3d_pass_orders.get((i, j), {})
                yield (
                    i,
                    j,
                    base_time,
                    travel_times.get('o1_to_o2_time'),
                    travel_times.get('o2_to_o1_time'),
                    bool(order_flags.get('o1o2', True)),
                    bool(order_flags.get('o2o1', True)),
                )

        success_count = 0
        failure_count = 0
        total_skipped_count = 0  # 统计跳过的已存在文件总数
        failure_reasons = Counter()
        tt_start_time = time.time()
        last_report_time = tt_start_time

        # 计算剩余任务数（从processed_tt_count之后）
        remaining_tasks = len(st3d_pass_pairs) - processed_tt_count
        print(f"[TT] 将从索引 {processed_tt_count} 开始处理，剩余 {remaining_tasks} 个任务")

        with ProcessPoolExecutor(max_workers=EVALUATION_WORKERS, initializer=_worker_initializer) as pool:
            inflight = set()
            task_iter = iter_tt_tasks()

            def submit_tt_next():
                """处理下一个TT任务并提交"""
                try:
                    (
                        i,
                        j,
                        base_time,
                        o1_to_o2_time,
                        o2_to_o1_time,
                        use_o1o2,
                        use_o2o1,
                    ) = next(task_iter)
                except StopIteration:
                    return False
                payload = {
                    'i': i,
                    'j': j,
                    't1': trip_list[i],
                    't2': trip_list[j],
                    'base_time': base_time,
                    'tt_dir': TT_OUTPUT_DIR,
                    'o1_to_o2_time': o1_to_o2_time,
                    'o2_to_o1_time': o2_to_o1_time,
                    'use_o1o2': use_o1o2,
                    'use_o2o1': use_o2o1,
                }
                fut = pool.submit(_evaluate_pair_worker, payload)
                inflight.add(fut)
                return True

            # 预填充任务
            while len(inflight) < min(max_inflight, remaining_tasks) and submit_tt_next():
                pass

            print(f"[TT] 初始化提交 {len(inflight)} 个任务，开始pymeshlab计算...")

            progress_total = remaining_tasks
            progress_step = 300000
            checkpoint_save_interval = 100000  # 每10万个任务保存一次检查点

            idx = 0
            while inflight:
                done = next(as_completed(inflight))
                inflight.remove(done)
                idx += 1
                try:
                    i, j, tt_count, pair_time, success, stage, reason, skipped_count = done.result()
                    if success and tt_count > 0:
                        success_count += tt_count
                    else:
                        failure_count += 1
                        failure_reasons[stage or 'tt'] += 1
                    total_skipped_count += skipped_count  # 累加跳过的文件数

                    # 计算当前已处理的总任务数（包括从检查点恢复的）
                    current_processed = processed_tt_count + idx

                    # 每完成 checkpoint_save_interval 个任务保存检查点
                    if current_processed % checkpoint_save_interval == 0:
                        checkpoint_mgr.save(
                            'tt',
                            {
                                'candidate_pairs': candidate_pairs,
                                'st3d_pass_pairs': st3d_pass_pairs,
                                'pair_travel_times': pair_travel_times,
                                'st3d_pass_orders': st3d_pass_orders,
                            },
                            {
                                'total_st3d_pairs': len(st3d_pass_pairs),
                                'processed_tt_count': current_processed,
                                'success_count': success_count,
                                'failure_count': failure_count,
                            },
                        )
                        print(f"[TT] 已保存检查点: 已处理 {current_processed}/{len(st3d_pass_pairs)} 个任务")

                    # 每完成progress_step个任务输出进度和耗时
                    if idx % progress_step == 0 or idx == progress_total:
                        current_time = time.time()
                        total_elapsed = current_time - tt_start_time
                        recent_elapsed = current_time - last_report_time
                        if idx % progress_step == 0:
                            # 正好完成progress_step个的倍数
                            recent_count = progress_step
                        else:
                            # 最后一批不足progress_step个，按实际数量
                            recent_count = idx % progress_step
                        print(
                            f"[进度] {current_processed}/{len(st3d_pass_pairs)} 完成，已成功 {success_count}，失败 {failure_count}，"
                            f"总耗时: {total_elapsed:.2f}秒，最近{recent_count}个耗时: {recent_elapsed:.2f}秒")
                        last_report_time = current_time

                except Exception as e:
                    failure_count += 1
                    print(f"[异常] Future执行失败: {e}")

                # 补充新的任务
                while len(inflight) < max_inflight and submit_tt_next():
                    pass

            # 在with块内保存最终检查点
            final_processed = processed_tt_count + idx
            checkpoint_mgr.save(
                'tt',
                {
                    'candidate_pairs': candidate_pairs,
                    'st3d_pass_pairs': st3d_pass_pairs,
                    'pair_travel_times': pair_travel_times,
                    'st3d_pass_orders': st3d_pass_orders,
                },
                {
                    'total_st3d_pairs': len(st3d_pass_pairs),
                    'processed_tt_count': final_processed,
                    'success_count': success_count,
                    'failure_count': failure_count,
                },
            )

        monitor.record_stage_end("TT计算（pymeshlab）")
        print(f"TT计算完成，成功 {success_count}，失败 {failure_count}，跳过已存在文件 {total_skipped_count}")

        if failure_reasons:
            print("失败阶段统计：")
            for stage, cnt in failure_reasons.most_common():
                print(f"  {stage}: {cnt}")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 发生异常时也尝试保存检查点（优先保存TT阶段，其次 st3d 和 candidate）
        try:
            if 'checkpoint_mgr' in locals():
                # 根据当前阶段保存检查点（合并模式会自动保留之前阶段的数据）
                if 'tt_started' in locals() and tt_started and 'st3d_pass_pairs' in locals() and len(
                        st3d_pass_pairs) > 0:
                    # 已经进入TT阶段，优先保存TT阶段检查点
                    current_processed = processed_tt_count if 'processed_tt_count' in locals() else 0
                    if 'idx' in locals():
                        current_processed += idx
                    checkpoint_mgr.save(
                        'tt',
                        {
                            'candidate_pairs': candidate_pairs if 'candidate_pairs' in locals() else [],
                            'st3d_pass_pairs': st3d_pass_pairs,
                            'pair_travel_times': pair_travel_times if 'pair_travel_times' in locals() else {},
                            'st3d_pass_orders': st3d_pass_orders if 'st3d_pass_orders' in locals() else {},
                        },
                        {
                            'total_st3d_pairs': len(st3d_pass_pairs),
                            'processed_tt_count': current_processed,
                            'error': str(e),
                            'timestamp': time.time(),
                        },
                    )
                elif 'st3d_pass_pairs' in locals() and len(st3d_pass_pairs) > 0:
                    checkpoint_mgr.save(
                        'st3d',
                        {
                            'st3d_pass_pairs': st3d_pass_pairs,
                            'pair_travel_times': pair_travel_times if 'pair_travel_times' in locals() else {},
                            'st3d_pass_orders': st3d_pass_orders if 'st3d_pass_orders' in locals() else {},
                        },
                        {'error': str(e), 'timestamp': time.time()},
                    )
                elif 'candidate_pairs' in locals() and len(candidate_pairs) > 0:
                    checkpoint_mgr.save('candidate', {
                        'candidate_pairs': candidate_pairs
                    }, {'error': str(e), 'timestamp': time.time()})
        except:
            pass

if __name__ == '__main__':
    main(
        checkpoint_name=None,
        resume=True,
        csv_path=r'D:\shirou\carpool\data\1375704000_30min.csv',
        start_ts=1375704000,
        slice_len=9645,
        max_trips=12000,
    )

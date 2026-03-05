# -*- coding: utf-8 -*-
import os
import re
import shutil
import time
import platform
import numpy as np
import trimesh
from typing import List, Dict, Optional, Tuple, Callable, TypedDict, Any
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.geometry.multipoint import MultiPoint

try:
    from shapely import speedups
except ImportError:
    speedups = None
from collections import Counter, defaultdict
import multiprocessing
from multiprocessing import Pool, shared_memory
from multiprocessing.shared_memory import SharedMemory
import tempfile
import uuid
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gc
import psutil
import psycopg2
from psycopg2 import pool
import sys

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

# ========== 全局缓存 ==========
_isochrone_points_cache = {}

# 主进程数据库连接池（用于主进程的批量查询）
_main_db_pool = None

# 棱柱缓存实例（单例）
_prism_cache = None


# 进程池 worker 全局状态类型定义
class WorkerStateDict(TypedDict):
    initialized: bool
    db_pool: Optional[pool.ThreadedConnectionPool]  # 每个worker进程内部的连接池


# 进程池 worker 全局状态（每个worker进程独立）
WORKER_STATE: WorkerStateDict = {
    'initialized': False,
    'db_pool': None  # 每个worker进程内部的连接池
}

# ========== 棱柱缓存配置 ==========
# 设置保存位置为 D:\shirou\carpool\分解\mesh
PRISM_CACHE_DIR = r'D:\shirou\carpool\分步结果数据\测试结果\节点mesh'
PRISM_CACHE_MAX_BYTES = 10 * 1024 * 1024 * 1024

# ========== 配置常量 ==========
CONFIG_MESH_WORKER_COUNT = multiprocessing.cpu_count()  # 默认使用CPU核心数
CONFIG_PRECACHE_PRISMS = True  # 是否启用棱柱预缓存
MAX_EXTRA_SECS = 300  # 乘客容忍的额外行驶时间上限（秒）


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
    批量查找多个点的最近节点（占位函数，需要根据实际需求实现）。

    Args:
        conn: 数据库连接
        points_list: 点列表，格式为 [(lon, lat), ...]

    Returns:
        节点列表，格式为 [(node_id, (lon, lat)), ...]
    """
    # TODO: 实现批量查找最近节点的逻辑
    # 这是一个占位函数，需要根据实际数据库结构实现
    results = []
    with conn.cursor() as cur:
        for lon, lat in points_list:
            sql = f"""
            SELECT id, ST_X(the_geom) as x, ST_Y(the_geom) as y
            FROM {VERTICES_TABLE}
            ORDER BY ST_Distance(
                the_geom,
                ST_SetSRID(ST_Point({lon}, {lat}), 4326)
            )
            LIMIT 1
            """
            cur.execute(sql)
            row = cur.fetchone()
            if row:
                results.append((row[0], (float(row[1]), float(row[2]))))
            else:
                results.append((None, None))
    return results


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


def _export_mesh_direct(mesh: trimesh.Trimesh, trip_id: int) -> Optional[str]:
    """
    将 mesh 直接导出到最终缓存位置，返回文件路径。
    如果目标文件已存在，则直接返回路径，不删除也不覆盖。
    如果失败返回 None。
    """
    dest_path = os.path.join(PRISM_CACHE_DIR, f"trip_{trip_id}.ply")

    # 如果目标文件已存在，直接返回路径，不删除也不覆盖
    if os.path.exists(dest_path):
        return dest_path

    # 确保缓存目录存在
    try:
        os.makedirs(PRISM_CACHE_DIR, exist_ok=True)
    except OSError as e:
        print(f"[导出mesh] trip_id={trip_id}: 创建缓存目录失败: {PRISM_CACHE_DIR}, 错误: {e}")
        return None

    # 优化：使用二进制PLY格式（更快）
    try:
        mesh.export(dest_path, file_type='ply')
    except Exception:
        # 回退到默认格式
        try:
            mesh.export(dest_path)
        except Exception as e:
            print(f"[导出mesh] trip_id={trip_id}: 导出mesh失败: {e}")
            return None

    return dest_path


class PrismCache:
    """简单的按行程ID缓存时空棱柱文件，带引用计数与LRU逐出。"""

    def __init__(self, cache_dir: str, max_bytes: int):
        self.cache_dir = cache_dir
        self.max_bytes = max_bytes
        self.entries: Dict[int, Dict] = {}
        self.total_size = 0
        self.failed_trip_ids = set()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _entry_path(self, trip_id: int) -> str:
        return os.path.join(self.cache_dir, f"trip_{trip_id}.ply")

    def checkout(self, trip_id: int) -> Optional[str]:
        entry = self.entries.get(trip_id)
        path = None

        if not entry:
            # 允许直接从磁盘发现已有文件（例如之前运行预缓存后进程重启）
            path = self._entry_path(trip_id)
            if not os.path.exists(path):
                return None
            try:
                size = os.path.getsize(path)
            except OSError:
                return None
            entry = {
                'path': path,
                'size': size,
                'last_used': time.time(),
                'ref_count': 0
            }
            self.entries[trip_id] = entry
            self.total_size += size
        else:
            path = entry['path']
            if not os.path.exists(path):
                self.total_size -= entry['size']
                self.entries.pop(trip_id, None)
                return None

        entry['ref_count'] += 1
        entry['last_used'] = time.time()
        return path

    def checkout_mesh(self, trip_id: int) -> Optional[trimesh.Trimesh]:
        """
        仅加载 mesh，不改变引用计数；调用方如果需要计数，应另行调用 checkout。
        这样与已有 _evaluate_pair_worker 的调用方式保持一致（先 load，再 checkout 增加引用）。
        """
        entry = self.entries.get(trip_id)
        path = None

        if not entry:
            # 与 checkout 一致，支持从磁盘发现已有文件
            path = self._entry_path(trip_id)
            if not os.path.exists(path):
                return None
            try:
                size = os.path.getsize(path)
            except OSError:
                return None
            entry = {
                'path': path,
                'size': size,
                'last_used': time.time(),
                'ref_count': 0
            }
            self.entries[trip_id] = entry
            self.total_size += size
        else:
            path = entry['path']
            if not os.path.exists(path):
                self.total_size -= entry['size']
                self.entries.pop(trip_id, None)
                return None
        try:
            mesh = trimesh.load(path, force='mesh')
            if mesh is None or mesh.is_empty:
                return None
            return mesh
        except Exception:
            return None

    def release(self, trip_id: int):
        entry = self.entries.get(trip_id)
        if not entry:
            return
        if entry['ref_count'] > 0:
            entry['ref_count'] -= 1

    def register_file(self, trip_id: int, file_path: str) -> Optional[str]:
        """
        注册已存在的文件到缓存（文件已经在最终位置，不需要移动）。
        用于 worker 进程直接保存到最终位置的情况。
        """
        if not file_path:
            print(f"[缓存注册] trip_id={trip_id}: 文件路径为空")
            return None
        if not os.path.exists(file_path):
            print(f"[缓存注册] trip_id={trip_id}: 文件不存在: {file_path}")
            return None

        # 删除旧缓存条目
        entry = self.entries.pop(trip_id, None)
        if entry:
            self.total_size -= entry['size']
            # 如果旧文件路径不同，尝试删除旧文件
            if entry['path'] != file_path and os.path.exists(entry['path']):
                try:
                    os.remove(entry['path'])
                except OSError:
                    pass

        try:
            size = os.path.getsize(file_path)
        except OSError as e:
            print(f"[缓存注册] trip_id={trip_id}: 获取文件大小失败: {file_path}, 错误: {e}")
            return None

        self.entries[trip_id] = {
            'path': file_path,
            'size': size,
            'last_used': time.time(),
            'ref_count': 0
        }
        self.total_size += size
        self._evict_if_needed()
        return file_path

    def _evict_if_needed(self):
        if self.total_size <= self.max_bytes:
            return
        # 仅逐出未被引用的条目
        eviction_candidates = sorted(
            ((trip_id, entry) for trip_id, entry in self.entries.items() if entry['ref_count'] == 0),
            key=lambda item: item[1]['last_used']
        )
        for trip_id, entry in eviction_candidates:
            if self.total_size <= self.max_bytes:
                break
            try:
                os.remove(entry['path'])
            except OSError:
                pass
            self.total_size -= entry['size']
            self.entries.pop(trip_id, None)

    def mark_failed(self, trip_id: int):
        self.failed_trip_ids.add(trip_id)

    def is_failed(self, trip_id: int) -> bool:
        return trip_id in self.failed_trip_ids

    def clear(self, remove_files: bool = False):
        """
        重置缓存记录，必要时一并删除磁盘文件。
        """
        for entry in list(self.entries.values()):
            if remove_files:
                try:
                    os.remove(entry['path'])
                except OSError:
                    pass
        self.entries.clear()
        self.failed_trip_ids.clear()
        self.total_size = 0
        if remove_files:
            try:
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            except Exception:
                pass


def get_prism_cache() -> PrismCache:
    global _prism_cache
    if _prism_cache is None:
        _prism_cache = PrismCache(PRISM_CACHE_DIR, PRISM_CACHE_MAX_BYTES)
    return _prism_cache


def isochrone_points(start_node_id, trip_time, direction='forward', conn=None):
    """
    生成等时圈点集，并对每层点集进行排序以便生成闭合曲线
    处理MULTILINESTRING几何类型
    Args:
        start_node_id: 起始节点ID
        trip_time: 总行程时间（秒）
        direction: 方向，'forward'表示从起点出发，'backward'表示从终点返回
    Returns:
        分层的等时圈点集，格式为 {layer_id: [points]}，每层点已排序
    """
    cache_key = (start_node_id, trip_time, direction)
    if cache_key in _isochrone_points_cache:
        return _isochrone_points_cache[cache_key]

    # # 重要：使用转换后的值替换原始变量，确保后续代码使用正确的类型
    # start_node_id = start_node_id_int
    # trip_time = trip_time_float

    close_conn = False
    if conn is None:
        conn = _get_worker_connection()
        close_conn = True
    result_points = {}
    try:
        # 设置方向相关的字段
        if direction == 'forward':
            cost_field = 'forward_travel_time'
            reverse_field = 'backward_travel_time'
        else:
            cost_field = 'backward_travel_time'
            reverse_field = 'forward_travel_time'
        # 均分时间间隔
        num_intervals = 12
        interval = trip_time / num_intervals
        time_thresholds = [i * interval for i in range(num_intervals + 1)]
        final_threshold_offset = trip_time - time_thresholds[-1]
        if direction == 'backward':
            time_thresholds = [t + final_threshold_offset for t in time_thresholds]
        for i in range(len(time_thresholds)):
            result_points[i] = []

        with conn.cursor() as cur:
            # 获取起点坐标
            sql_start_coords = f"""
            SELECT ST_X(the_geom) as x, ST_Y(the_geom) as y
            FROM {VERTICES_TABLE}
            WHERE id = {start_node_id}
            """
            cur.execute(sql_start_coords)
            start_coords = cur.fetchone()
            if not start_coords:
                return {}
            start_x, start_y = float(start_coords[0]), float(start_coords[1])

            # 计算搜索半径（基于时间和平均速度）
            # 假设平均速度40km/h，转换为度/秒
            avg_speed_deg_per_sec = 40 / 111000  # 40km/h ≈ 40/111000 度/秒
            search_radius_deg = (trip_time + 30) * avg_speed_deg_per_sec * 2  # 2倍安全系数

            sql_driving_distance = f"""
            SELECT node, agg_cost
            FROM pgr_drivingDistance(
                'SELECT gid as id, source, target, {cost_field} as cost, {reverse_field} as reverse_cost 
                 FROM {ROAD_TABLE} 
                 WHERE ST_DWithin(geom, ST_SetSRID(ST_Point({start_x}, {start_y}), 4326), {search_radius_deg})',
                {start_node_id},
                {trip_time + 30},
                true
            )
            """
            try:
                cur.execute(sql_driving_distance)
                node_costs = cur.fetchall()
            except psycopg2.errors.OutOfMemory as e:
                # PostgreSQL内存用尽，返回空结果
                error_msg = f"生成等时圈点集时出错: PostgreSQL内存用尽 (start_node_id={start_node_id}, trip_time={trip_time})\n"
                error_msg += f"DETAIL: {str(e)}\n"
                sys.stderr.write(error_msg)
                sys.stderr.flush()
                return {}
            except Exception as e:
                # 其他数据库错误
                error_msg = f"生成等时圈点集时数据库错误 (start_node_id={start_node_id}, trip_time={trip_time}): {str(e)}\n"
                sys.stderr.write(error_msg)
                sys.stderr.flush()
                return {}
            cost_map = {int(row[0]): float(row[1]) for row in node_costs}
            if not cost_map:
                return {}
            node_ids = list(cost_map.keys())
            sql_nodes = f"""
            SELECT id, ST_X(the_geom) as x, ST_Y(the_geom) as y
            FROM {VERTICES_TABLE}
            WHERE id IN ({','.join(map(str, node_ids))})
            """
            cur.execute(sql_nodes)
            node_coords = cur.fetchall()
            node_coord_map = {row[0]: (float(row[1]), float(row[2])) for row in node_coords}
            sql_start_node = f"""
            SELECT ST_X(the_geom) as x, ST_Y(the_geom) as y
            FROM {VERTICES_TABLE}
            WHERE id = {start_node_id}
            """
            cur.execute(sql_start_node)
            start_coords = cur.fetchone()
            if not start_coords:
                return {}
            start_x, start_y = float(start_coords[0]), float(start_coords[1])
            # 批量处理节点分配
            node_id_list = [int(nid) for nid in cost_map.keys()]
            node_ids_array = np.array(node_id_list, dtype=np.int64)
            costs_array = np.array([float(cost_map[nid]) for nid in node_id_list], dtype=np.float64)
            thresholds_array = np.array(time_thresholds, dtype=np.float64)
            node_coords_arr = np.array([node_coord_map[nid] for nid in node_id_list], dtype=np.float64)
            angles = np.arctan2(node_coords_arr[:, 1] - start_y, node_coords_arr[:, 0] - start_x)
            relative_tolerance = interval * 1e-6
            for i, threshold in enumerate(thresholds_array):
                mask = np.abs(costs_array - threshold) < relative_tolerance
                matching_nodes = node_ids_array[mask]
                matching_costs = costs_array[mask]
                matching_coords = node_coords_arr[mask]
                matching_angles = angles[mask]
                for j, node_id in enumerate(matching_nodes):
                    tobreak = threshold if direction == 'forward' else (trip_time - threshold)
                    result_points[i].append({
                        'x': matching_coords[j, 0],
                        'y': matching_coords[j, 1],
                        'z': threshold,
                        'node_id': int(node_id),
                        'type': 'network_node',
                        'agg_cost': matching_costs[j],
                        'angle': matching_angles[j],
                        'tobreak': tobreak
                    })
            # 查询所有边
            sql_edges = f"""
            SELECT e.gid, e.source, e.target, e.{cost_field} as cost,
                    ST_X(ST_StartPoint(d.geom)) as start_x,
                    ST_Y(ST_StartPoint(d.geom)) as start_y,
                    ST_X(ST_EndPoint(d.geom)) as end_x,
                    ST_Y(ST_EndPoint(d.geom)) as end_y,
                    ST_Length(ST_Transform(d.geom, 3857)) as length
            FROM {ROAD_TABLE} e,
            LATERAL ST_Dump(e.geom) AS d
            WHERE (e.source IN ({','.join(map(str, node_ids))}) OR e.target IN ({','.join(map(str, node_ids))}));
            """

            cur.execute(sql_edges)
            edges = cur.fetchall()
            for edge in edges:
                try:
                    edge_id, source, target, edge_cost = edge[:4]
                    if len(edge) < 8 or any(coord is None for coord in edge[4:8]):
                        continue
                    start_x, start_y, end_x, end_y = edge[4:8]
                    edge_length = edge[8] if len(edge) > 8 else None
                    source_cost = cost_map.get(source)
                    target_cost = cost_map.get(target)
                    if source_cost is None and target_cost is None:
                        continue
                    if source_cost is not None and target_cost is not None and source_cost > target_cost:
                        source, target = target, source
                        source_cost, target_cost = target_cost, source_cost
                        start_x, end_x = end_x, start_x
                        start_y, end_y = end_y, start_y
                    for i, threshold in enumerate(time_thresholds):
                        if i == 0:
                            continue
                        if (source_cost is not None and target_cost is not None and
                                source_cost <= threshold <= target_cost):
                            lambda_val = (threshold - source_cost) / (target_cost - source_cost)
                            x = start_x + lambda_val * (end_x - start_x)
                            y = start_y + lambda_val * (end_y - start_y)
                            angle = np.arctan2(y - start_y, x - start_x)
                            tobreak = threshold if direction == 'forward' else (trip_time - threshold)
                            result_points[i].append({
                                'x': x,
                                'y': y,
                                'z': threshold,
                                'edge_id': edge_id,
                                'source': source,
                                'target': target,
                                'lambda': lambda_val,
                                'type': 'interpolated',
                                'agg_cost': threshold,
                                'edge_length': edge_length,
                                'angle': angle,
                                'tobreak': tobreak
                            })
                except Exception as e:
                    continue
            if not result_points[0] and start_node_id in node_coord_map:
                x, y = node_coord_map[start_node_id]
                tobreak = 0.0 if direction == 'forward' else trip_time
                result_points[0].append({
                    'x': x,
                    'y': y,
                    'z': 0,
                    'node_id': start_node_id,
                    'type': 'network_node',
                    'agg_cost': 0,
                    'angle': 0,
                    'tobreak': tobreak
                })
        for layer in result_points:
            if result_points[layer]:
                result_points[layer].sort(key=lambda p: p['angle'])
        _isochrone_points_cache[cache_key] = result_points
        return result_points
    except Exception as e:
        import traceback
        error_msg = f"生成等时圈点集时出错 (start_node_id={start_node_id}, trip_time={trip_time}, direction={direction}): {str(e)}\n"
        error_msg += traceback.format_exc()
        sys.stderr.write(error_msg)
        sys.stderr.flush()
        return {}
    finally:
        if close_conn and conn is not None:
            _return_worker_connection(conn)


def vertical_projection_intersect_z(point_xy: Tuple[float, float], mesh: trimesh.Trimesh) -> Optional[List[float]]:
    """
    给定一个 XY 点，计算其在网格上的垂直投影交点的 z 值范围。
    """
    # x0, y0 = point_xy
    # 确保point_xy是元组或列表，并且值是float类型（处理Decimal等类型）
    if not isinstance(point_xy, (tuple, list)) or len(point_xy) < 2:
        return None
    try:
        x0 = float(point_xy[0])
        y0 = float(point_xy[1])
    except (ValueError, TypeError):
        return None
    vertices = mesh.vertices
    faces = mesh.faces
    z_list = []

    for face in faces:
        v0, v1, v2 = vertices[face]
        x1, y1 = v0[0], v0[1]
        x2, y2 = v1[0], v1[1]
        x3, y3 = v2[0], v2[1]

        # 计算三角形重心系数（barycentric coordinates）
        detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(detT) < 1e-10:
            continue  # 跳过退化面

        l1 = ((y2 - y3) * (x0 - x3) + (x3 - x2) * (y0 - y3)) / detT
        l2 = ((y3 - y1) * (x0 - x3) + (x1 - x3) * (y0 - y3)) / detT
        l3 = 1 - l1 - l2

        # 容差范围内认为在三角形内
        if (l1 >= -1e-8) and (l2 >= -1e-8) and (l3 >= -1e-8):
            z = l1 * v0[2] + l2 * v1[2] + l3 * v2[2]
            z_list.append(z)

    if z_list:
        return [min(z_list), max(z_list)]
    else:
        return None


def create_closed_linestrings_in_memory(points_dict):
    """
    仿照原create_closed_linestrings，返回{layer_id: Polygon}
    完全复制原始逻辑，包括去重、第0层特殊处理、角度计算等
    使用NumPy批量处理优化性能
    """
    layer_polygons = {}
    for layer_id, points in points_dict.items():
        # 移除重复点（基于坐标）- 完全复制原始逻辑
        unique_points = []
        seen_coords = set()
        for p in points:
            coord = (p['x'], p['y'])
            if coord not in seen_coords:
                unique_points.append(p)
                seen_coords.add(coord)

        # 特殊处理第0层（起点/终点）- 完全复制原始逻辑
        if layer_id == 0:
            if len(unique_points) >= 1:
                # 第0层只使用一个点（通常是起点/终点）
                origin_point = unique_points[0]
                # 创建一个小圆作为起点/终点的表示
                # 在点周围创建一个小圆环（半径约10米）
                radius = 0.0001  # 约10米的经纬度差
                # 使用NumPy批量生成圆环点
                angles = np.arange(0, 360, 10) * (np.pi / 180)  # 每10度一个点
                x_coords = origin_point['x'] + radius * np.cos(angles)
                y_coords = origin_point['y'] + radius * np.sin(angles)
                circle_points = list(zip(x_coords, y_coords))

                # 闭合圆环
                circle_points.append(circle_points[0])

                # 创建Polygon对象
                poly = Polygon(circle_points)
                if poly.is_valid and not poly.is_empty:
                    layer_polygons[layer_id] = poly
            continue

        # 处理其他层
        # 跳过没有足够点的层
        if len(unique_points) < 3:
            continue

        # 使用NumPy批量计算中心点
        coords = np.array([[p['x'], p['y']] for p in unique_points])
        center = coords.mean(axis=0)
        center_x, center_y = center

        # 使用NumPy批量计算极坐标角度和距离
        dx = coords[:, 0] - center_x
        dy = coords[:, 1] - center_y
        angles = np.arctan2(dy, dx)
        # 转换为0-360度
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)
        # distances = np.sqrt(dx ** 2 + dy ** 2)

        # 将计算结果赋值给每个点
        # for i, p in enumerate(unique_points):
        #     p['polar_angle'] = angles[i]
        #     p['center_dist'] = distances[i]
        # 用 NumPy 排序索引,稳定排序 (mergesort)
        sorted_indices = np.argsort(angles, kind="mergesort")
        sorted_coords = coords[sorted_indices]
        # 闭合环
        sorted_coords = np.vstack([sorted_coords, sorted_coords[0]])
        # 按极坐标角度排序
        # sorted_points = sorted(unique_points, key=lambda p: p['polar_angle'])

        # # 为每个点分配group_no（从1开始）
        # for i, p in enumerate(sorted_points):
        #     p['group_no'] = i + 1
        #
        # # 提取坐标
        # coordinates = [(p['x'], p['y']) for p in sorted_points]
        #
        # # 闭合线环（添加第一个点作为最后一个点）
        # coordinates.append(coordinates[0])

        # 创建Polygon对象
        poly = Polygon(sorted_coords)
        if poly.is_valid and not poly.is_empty:
            layer_polygons[layer_id] = poly

    return layer_polygons


def calculate_line_intersection_in_memory(origin_polys, dest_polys, tobreaks_origin, tobreaks_dest, trip_time=None):
    """
    用tobreak值作为key进行匹配，仅保留真正属于两边可达集交集的时间层。
    origin_polys, dest_polys: {layer_id: Polygon}
    tobreaks_origin, tobreaks_dest: {layer_id: tobreak}
    输出: {rounded_tobreak: Polygon}
    """

    # 构建tobreak到Polygon的映射（四舍五入保留6位小数以解决浮点数精度问题）
    def build_tobreak_map(polys, tobreaks):
        tobreak_map = {}
        for layer_id, poly in polys.items():
            tobreak_value = tobreaks[layer_id]
            # 确保tobreak_value是Python原生float类型，而不是numpy数组
            try:
                if isinstance(tobreak_value, np.ndarray):
                    tobreak_float = float(tobreak_value.flat[0]) if tobreak_value.size > 0 else 0.0
                elif hasattr(tobreak_value, 'item'):
                    tobreak_float = float(tobreak_value.item())
                elif isinstance(tobreak_value, (list, tuple)):
                    tobreak_float = float(tobreak_value[0]) if len(tobreak_value) > 0 else 0.0
                else:
                    tobreak_float = float(tobreak_value)
            except (ValueError, TypeError, IndexError):
                tobreak_float = float(tobreak_value) if tobreak_value is not None else 0.0

            rounded = round(tobreak_float, 6)  # 四舍五入保留6位小数
            tobreak_map[rounded] = poly
        return tobreak_map

    origin_map = build_tobreak_map(origin_polys, tobreaks_origin)
    dest_map = build_tobreak_map(dest_polys, tobreaks_dest)

    common_tobreaks = set(origin_map.keys()) & set(dest_map.keys())

    intersection_map = {}
    empty_intersections = 0
    for t in sorted(common_tobreaks):
        poly1 = origin_map[t]
        poly2 = dest_map[t]
        inter = poly1.intersection(poly2)
        if inter.is_empty or not inter.is_valid:
            empty_intersections += 1
            continue
        if inter.geom_type == 'Polygon':
            intersection_map[t] = inter
        elif inter.geom_type == 'MultiPolygon':
            # print(f"tobreak={t}: 交集有MultiPolygon，取最大面")
            largest = max(inter.geoms, key=lambda g: g.area)
            intersection_map[t] = largest

    # 此处不再对起点层(t=0.0)和终点层(t=trip_time)做任何强制加入处理，
    # 只保留真正属于两边可达集交集的时间层。

    return intersection_map


def create_3d_prism_from_intersections_in_memory(intersection_polys, tobreaks=None, connection_method='convex_hull'):
    """
    输入: {layer_id: Polygon}, tobreaks: list or None
    输出: trimesh.Trimesh (水密mesh) 或 None (如果交集退化)
    """
    if not intersection_polys:
        print("警告: 没有交集多边形")
        return None

    # 按layer_id排序
    layers = [intersection_polys[k] for k in sorted(intersection_polys.keys())]
    if tobreaks is None:
        tobreaks = list(sorted(intersection_polys.keys()))

    # 检查是否有足够的层来构建3D棱柱
    if len(layers) < 2:
        # print("警告: 交集层数不足，无法构建3D棱柱")
        return None

    # 检查tobreaks是否有变化（确保有3D体积）
    if len(set(tobreaks)) < 2:
        # print("警告: 所有层的tobreak值相同，交集退化为一维")
        return None

    # 直接对所有3D点做ConvexHull
    num_points = 54
    resampled_layers = []
    for poly in layers:
        coords = np.array(poly.exterior.coords)
        perimeter = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
        dists = np.cumsum([0] + [np.linalg.norm(coords[i] - coords[i - 1]) for i in range(1, len(coords))])
        dists = dists / dists[-1]
        new_coords = []
        for i in range(num_points):
            t = i / num_points
            idx = np.searchsorted(dists, t, side='right') - 1
            alpha = (t - dists[idx]) / (dists[idx + 1] - dists[idx]) if dists[idx + 1] > dists[idx] else 0
            pt = coords[idx] + alpha * (coords[idx + 1] - coords[idx])
            new_coords.append(pt)
        resampled_layers.append(new_coords)

    # 生成所有3D点
    vertices_3d = []
    for i, layer in enumerate(resampled_layers):
        z = tobreaks[i]
        for x, y in layer:
            vertices_3d.append([x, y, z])
    vertices_3d = np.array(vertices_3d)

    # 检查3D点的维度
    if len(vertices_3d) < 4:
        # print("警告: 3D顶点数量不足，无法构建凸包")
        return None

    # 检查z坐标是否有变化
    z_coords = vertices_3d[:, 2]
    if np.std(z_coords) < 1e-10:  # 如果z坐标几乎相同
        # print("警告: 所有点的z坐标相同，交集退化为一维")
        return None

    try:
        # 直接做3D凸包
        hull = ConvexHull(vertices_3d)
        faces = hull.simplices.tolist()
        mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces)
        return mesh
    except Exception as e:
        print(f"构建3D凸包失败: {str(e)}")
        print("这表明交集退化，拼车方案不可行")
        return None


def build_spacetime_prism_from_pgrouting(trip_id: int, start_node_id: int, end_node_id: int, trip_time: float,
                                         time_min, point_type: str = '', conn=None) -> Tuple[
    Optional[trimesh.Trimesh], Optional[str]]:
    """
    部分内存版：交集计算在内存中进行，但等时圈生成仍依赖数据库

    已内存化的部分：
    - create_closed_linestrings_in_memory: 生成Polygon而非数据库表
    - calculate_line_intersection_in_memory: 内存中计算Polygon交集
    - create_3d_prism_from_intersections_in_memory: 内存中生成3D网格

    仍依赖数据库的部分：
    - isochrone_points_fixed_interval: pgRouting等时圈生成
    """
    close_conn = False
    if conn is None:
        conn = _get_worker_connection()
        close_conn = True
    try:
        # 直接使用传入的节点ID，无需重复查找
        if not start_node_id or not end_node_id:
            error_msg = f"无法找到行程 {trip_id} 的起点或终点的节点ID"
            return None, error_msg

        origin_points = isochrone_points(start_node_id, trip_time, direction="forward", conn=conn)
        if not origin_points:
            error_msg = f"trip_id={trip_id}: 起点等时圈为空 (start_node_id={start_node_id}, trip_time={trip_time})"
            return None, error_msg

        destination_points = isochrone_points(end_node_id, trip_time, direction="backward", conn=conn)
        if not destination_points:
            error_msg = f"trip_id={trip_id}: 终点等时圈为空 (end_node_id={end_node_id}, trip_time={trip_time})"
            return None, error_msg

        # 生成密闭线（Polygon）
        origin_polys = create_closed_linestrings_in_memory(origin_points)
        destination_polys = create_closed_linestrings_in_memory(destination_points)

        if not origin_polys:
            error_msg = f"trip_id={trip_id}: 起点多边形为空 (起点等时圈层数={len(origin_points)})"
            return None, error_msg

        if not destination_polys:
            error_msg = f"trip_id={trip_id}: 终点多边形为空 (终点等时圈层数={len(destination_points)})"
            return None, error_msg

        # 提取每层的tobreak值
        tobreaks_origin = {layer_id: points[0]['tobreak'] for layer_id, points in origin_points.items() if points}
        tobreaks_dest = {layer_id: points[0]['tobreak'] for layer_id, points in destination_points.items() if points}
        # 计算交集（仅保留真正属于两边可达集交集的时间层）
        intersection_polys = calculate_line_intersection_in_memory(
            origin_polys, destination_polys, tobreaks_origin, tobreaks_dest, trip_time=trip_time
        )

        if not intersection_polys:
            error_msg = f"trip_id={trip_id}: 交集多边形为空 (起点多边形层数={len(origin_polys)}, 终点多边形层数={len(destination_polys)})"
            return None, error_msg

        # 额外约束：起点和终点自身也必须同时属于两边可达集
        # 起点对应 t=0.0（经过 round(6) 之后依然是 0.0）
        # start_t = 0.0

        # # 终点对应 t=trip_time（需与 calculate_line_intersection_in_memory 中的 round(6) 一致）
        # try:
        #     if isinstance(trip_time, np.ndarray):
        #         trip_time_float = float(trip_time.flat[0]) if trip_time.size > 0 else 0.0
        #     elif hasattr(trip_time, 'item'):
        #         trip_time_float = float(trip_time.item())
        #     elif isinstance(trip_time, (list, tuple)):
        #         trip_time_float = float(trip_time[0]) if len(trip_time) > 0 else 0.0
        #     else:
        #         trip_time_float = float(trip_time)
        # except (ValueError, TypeError, IndexError):
        #     trip_time_float = float(trip_time) if trip_time is not None else 0.0

        # end_t = round(trip_time_float, 6)

        # if start_t not in intersection_polys or end_t not in intersection_polys:
        #     error_msg = (
        #         f"trip_id={trip_id}: 起点层或终点层在OD交集中缺失，"
        #         f"起点层存在={start_t in intersection_polys}, 终点层存在={end_t in intersection_polys}"
        #     )
        #     return None, error_msg

        # 生成3D棱柱mesh
        tobreaks = list(sorted(intersection_polys.keys()))
        mesh = create_3d_prism_from_intersections_in_memory(intersection_polys, tobreaks,
                                                            connection_method='convex_hull')

        if mesh is None:
            error_msg = f"trip_id={trip_id}: 生成3D棱柱mesh失败 (交集层数={len(intersection_polys)}, tobreaks={len(tobreaks)})"
            return None, error_msg

        # ================== 网格重修复前移 ==================
        # 此处执行一次较重的网格清理/修复，保证后续 TT 布尔交集阶段输入质量，
        # 避免在 CarpoolFeasibilityChecker.compute_mesh_intersection 中重复对同一棱柱多次修复。
        # try:
        #     # 基本处理：去除退化面、重复面等
        #     mesh.process(validate=False)
        #     # 移除重复三角面
        #     mesh.update_faces(mesh.unique_faces())
        #     # 去掉退化三角形（面积为0）
        #     mesh.update_faces(mesh.nondegenerate_faces())
        #     # 修复法线
        #     mesh.fix_normals()
        #     # 填充小孔
        #     mesh.fill_holes()
        # except Exception as e:
        #     # 修复失败时仅打印告警，不中断主流程
        #     print(f"[棱柱预修复警告] trip_id={trip_id}, point_type={point_type}, 错误={e}")

        mesh.apply_translation([0, 0, time_min])

        return mesh, None

    except Exception as e:
        import traceback
        error_msg = f"构建时空棱柱时异常: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg
    finally:
        if close_conn and conn is not None:
            _return_worker_connection(conn)


def _build_prism_and_dump(build_args: Tuple, use_shared_memory: bool = True) -> Tuple[Optional[Dict], Optional[str]]:
    """
    动态任务：构建单个行程的时空棱柱，使用共享内存传递（优化版本）。

    Args:
        build_args: 构建参数元组
        use_shared_memory: 是否使用共享内存（默认True），False时回退到文件方式
        注意：在Windows系统上，共享内存会自动禁用，使用文件方式

    Returns:
        (result, error_message) 元组:
        - result: 如果成功，返回共享内存元数据字典或文件路径字符串；如果失败，返回None
        - error_message: 如果失败，返回错误信息；如果成功，返回None
    """
    trip_id = build_args[0] if build_args else 'unknown'
    # _log_worker_task('build_prism', f"trip_id={trip_id}")

    # 检查目标文件是否已存在，如果存在则跳过构建
    dest_path = os.path.join(PRISM_CACHE_DIR, f"trip_{trip_id}.ply")
    if os.path.exists(dest_path):
        # 文件已存在，直接返回文件路径，跳过构建
        return dest_path, None

    conn = None
    error_message = None
    try:
        conn = _get_worker_connection()
        mesh, error_message = build_spacetime_prism_from_pgrouting(*build_args, conn=conn)
        if mesh is None:
            # 构建失败，error_message已经包含了具体的失败原因
            return None, error_message
        # 直接保存到最终位置
        result = _export_mesh_direct(mesh, trip_id)
        if result is None:
            error_message = f"trip_id={trip_id}: 导出mesh到文件失败"
            return None, error_message
        return result, None
    except Exception as e:
        import traceback
        error_message = f"trip_id={trip_id}: 构建时空棱柱异常 - {str(e)}\n{traceback.format_exc()}"
        return None, error_message
    finally:
        if conn is not None:
            _return_worker_connection(conn)


def precache_all_trips(trip_list, mesh_worker_count=None, progress_callback=None):
    """
    为所有单独行程构建并缓存棱柱（不依赖行程对）

    Args:
        trip_list: 行程列表
        mesh_worker_count: Mesh生成进程数，None时使用配置值
        progress_callback: 进度回调函数，接收 (completed, total) 参数

    Returns:
        (success_count, failed_count, failed_trip_ids)
    """
    if not trip_list:
        return 0, 0, []

    if mesh_worker_count is None:
        mesh_worker_count = CONFIG_MESH_WORKER_COUNT

    # 所有行程索引
    all_trip_indices = list(range(len(trip_list)))
    total_prisms = len(all_trip_indices)

    print(f"[棱柱预缓存] 开始预构建 {total_prisms} 个行程的棱柱...")
    print(f"[棱柱预缓存] 使用 {mesh_worker_count} 个Mesh worker进程")
    print(f"[棱柱预缓存] 缓存目录: {PRISM_CACHE_DIR}")

    prism_cache = get_prism_cache()

    # 准备构建参数
    trip_build_args = {}
    trip_ids_to_build = []

    for trip_idx in all_trip_indices:
        trip = trip_list[trip_idx]
        trip_id = trip['trip_id']

        # 检查缓存是否已存在
        cached_path = prism_cache.checkout(trip_id)
        if cached_path:
            # 已存在，释放引用计数
            prism_cache.release(trip_id)
            continue

        # 检查是否已标记为失败
        if prism_cache.is_failed(trip_id):
            continue

        # 需要构建
        trip_build_args[trip_id] = (
            trip_id,
            trip['O_node_id'],
            trip['D_node_id'],
            trip.get('max_trip_time', trip['trip_time']),
            0.0,
            'trip'
        )
        trip_ids_to_build.append((trip_id, trip_idx))

    if not trip_ids_to_build:
        print(f"[棱柱预缓存] 所有 {total_prisms} 个棱柱已存在于缓存中，跳过构建")
        return total_prisms, 0, []

    print(f"[棱柱预缓存] 需要构建 {len(trip_ids_to_build)} 个新棱柱（{total_prisms - len(trip_ids_to_build)} 个已缓存）")

    # 使用进程池并行构建
    success_count = 0
    failed_count = 0
    failed_trip_ids = []

    start_time = time.time()
    last_progress_time = start_time
    progress_interval = 5.0  # 每5秒打印一次进度

    with ProcessPoolExecutor(max_workers=mesh_worker_count, initializer=_worker_initializer) as mesh_pool:
        # 提交所有构建任务
        futures = {}
        for trip_id, trip_idx in trip_ids_to_build:
            build_args = trip_build_args[trip_id]
            try:
                fut = mesh_pool.submit(_build_prism_and_dump, build_args)
                futures[fut] = (trip_id, trip_idx)
            except Exception as e:
                print(f"[棱柱预缓存] 提交任务失败 trip_id={trip_id}: {e}")
                failed_count += 1
                failed_trip_ids.append(trip_id)
                prism_cache.mark_failed(trip_id)

        # 等待所有任务完成
        completed = 0
        total_futures = len(futures)

        for fut in as_completed(futures, timeout=None):
            trip_id, trip_idx = futures[fut]
            try:
                result, error_message = fut.result(timeout=0)  # as_completed已经等待完成，无需再等待

                if result is None:
                    # 构建失败
                    failed_count += 1
                    failed_trip_ids.append(trip_id)
                    prism_cache.mark_failed(trip_id)
                    # if error_message:
                    #     print(f"[棱柱预缓存] 构建失败 trip_id={trip_id} (trip_idx={trip_idx}): {error_message}")
                    # else:
                    #     print(f"[棱柱预缓存] 构建失败 trip_id={trip_id} (trip_idx={trip_idx}): 未知原因")
                else:
                    # 构建成功，注册到缓存（文件已经在最终位置）
                    if isinstance(result, str):
                        # 文件路径（已经在最终位置）
                        cached_path = prism_cache.register_file(trip_id, result)
                        if cached_path:
                            success_count += 1
                        else:
                            failed_count += 1
                            failed_trip_ids.append(trip_id)
                            prism_cache.mark_failed(trip_id)
                            print(f"[棱柱预缓存] 注册失败 trip_id={trip_id}: 无法注册到缓存")
                    else:
                        print(f"[棱柱预缓存] 未知结果类型 trip_id={trip_id}: {type(result)}")
                        failed_count += 1
                        failed_trip_ids.append(trip_id)

                completed += 1

                # 进度报告
                current_time = time.time()
                if current_time - last_progress_time >= progress_interval or completed == total_futures:
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = total_futures - completed
                    eta = remaining / rate if rate > 0 else 0

                    print(f"[棱柱预缓存] 进度: {completed}/{total_futures} "
                          f"({100 * completed / total_futures:.1f}%), "
                          f"成功: {success_count}, 失败: {failed_count}")

                    if progress_callback:
                        try:
                            progress_callback(completed, total_futures)
                        except Exception:
                            pass

                    last_progress_time = current_time

            except Exception as e:
                failed_count += 1
                failed_trip_ids.append(trip_id)
                prism_cache.mark_failed(trip_id)
                print(f"[棱柱预缓存] 构建异常 trip_id={trip_id} (trip_idx={trip_idx}): {e}")
                completed += 1

    elapsed_time = time.time() - start_time
    print(f"[棱柱预缓存] 完成！总用时: {elapsed_time:.2f}秒")
    print(f"[棱柱预缓存] 成功: {success_count}, 失败: {failed_count}, "
          f"平均速度: {success_count / elapsed_time:.2f} 个/秒" if elapsed_time > 0 else "")

    return success_count, failed_count, failed_trip_ids


def main():
    """批量处理主函数（使用进程池优化）"""
    # 开始性能监控
    monitor = start_monitoring()

    try:
        start_time0 = time.time()

        # 1. 批量读取CSV
        monitor.record_stage_start("数据读取")
        start_time_read = time.time()
        # csv_path = r'D:\shirou\carpool\data\setpoins.csv'
        # df = pd.read_csv(csv_path)
        # print(f"读取CSV文件，共 {len(df)} 行")
        # csv_path = r'D:\shirou\carpool\data\mm_trips_2013-08_snaped_renamed.csv'
        csv_path = r'D:\shirou\carpool\data\1375704000_30min.csv'
        # csv_path = r'D:\shirou\carpool\data\1375740000_30min.csv'
        df = pd.read_csv(csv_path)
        start_ts = 1375704000
        df = df.sort_values('pickup_datetime').reset_index(drop=True)
        valid_idx = df.index[df['pickup_datetime'] >= start_ts]
        if len(valid_idx) == 0:
            df = df.iloc[0:0].copy()
        else:
            start_pos = int(valid_idx[0])
            # end_pos = start_pos + 11813
            end_pos = start_pos + 9645
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

        max_trips = 15000  # 可调整
        df = df.head(max_trips).reset_index(drop=True)
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
        max_extra_secs = MAX_EXTRA_SECS  # 乘客容忍的额外行驶时间上限（秒）
        for idx, row in df.iterrows():
            O = (row['pickup_longitude'], row['pickup_latitude'])
            D = (row['dropoff_longitude'], row['dropoff_latitude'])
            trip_time = float(row['trip_time_in_secs'])
            start_time = int(row['pickup_timestamp'])
            O_node_id, O_node = pickup_nodes[idx]
            D_node_id, D_node = dropoff_nodes[idx]

            # 计算包含容忍度的最大行程时间
            extra_time = min(max_extra_secs, 0.15 * trip_time)
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

        # ---------- 3. 为所有单独行程生成棱柱并保存 ----------
        if CONFIG_PRECACHE_PRISMS and len(trip_list) > 0:
            print("=" * 60)
            print(f"开始为所有 {len(trip_list)} 个行程生成棱柱并保存到缓存...")
            print(f"缓存目录: {PRISM_CACHE_DIR}")
            print(f"最大缓存大小: {PRISM_CACHE_MAX_BYTES / (1024 ** 3):.1f} GB")
            monitor.record_stage_start("棱柱预缓存")
            precache_start_time = time.time()

            success_count, failed_count, failed_trip_ids = precache_all_trips(
                trip_list=trip_list,
                mesh_worker_count=CONFIG_MESH_WORKER_COUNT
            )

            precache_end_time = time.time()
            monitor.record_stage_end("棱柱预缓存")
            print(f"[棱柱预缓存] 总耗时: {precache_end_time - precache_start_time:.2f}秒")

            if failed_count > 0:
                print(f"[警告] 有 {failed_count} 个棱柱构建失败")
                if len(failed_trip_ids) <= 20:
                    print(f"[警告] 失败的行程ID: {failed_trip_ids}")
                else:
                    print(f"[警告] 失败的行程ID（前20个）: {failed_trip_ids[:20]}...")
            print("=" * 60)
        else:
            if not CONFIG_PRECACHE_PRISMS:
                print("[信息] 棱柱预缓存已禁用，将按需构建棱柱")
            elif len(trip_list) == 0:
                print("[信息] 没有行程数据，跳过棱柱预缓存")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

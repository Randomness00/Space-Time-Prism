# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import os
from scipy.interpolate import griddata
from matplotlib.ticker import FuncFormatter
import trimesh
from scipy.spatial import Delaunay
import json
from pylab import mpl
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号

def read_ordered_points_from_postgres(connection_string, table_name):
    """从PostgreSQL数据库读取原始数据"""
    points_dict = {}
    query = f"""
        SELECT 
            "FID_Servic", 
            "GroupNo", 
            ST_X(geometry) as x, 
            ST_Y(geometry) as y, 
            "ToBreak" as z
        FROM {table_name}
    """

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            for row in cursor.fetchall():
                fid, group_no, x, y, z = row
                x = float(x) if x is not None else 0.0
                y = float(y) if y is not None else 0.0
                z = float(z) if z is not None else 0.0

                if fid not in points_dict:
                    points_dict[fid] = []

                points_dict[fid].append({
                    "group_no": group_no,
                    "x": x,
                    "y": y,
                    "z": z
                })

    return points_dict

def create_interpolation_function(points_dict):
    """创建二维插值函数"""
    points = []
    for fid in points_dict:
        for p in points_dict[fid]:
            points.append([p['x'], p['y'], p['z']])
    points = np.array(points)

    # 创建插值函数
    def interpolate(x, y):
        # 使用最近邻插值
        points_2d = points[:, :2]
        values = points[:, 2]
        return griddata(points_2d, values, (x, y), method='nearest')
    return interpolate

def resample_points(points, num_points):
    """重采样点集，确保每层点数一致"""
    if len(points) <= num_points:
        return points

    # 计算累积距离
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i-1] + np.linalg.norm(points[i][:2] - points[i-1][:2])

    # 创建均匀分布的参数
    t = np.linspace(0, distances[-1], num_points)

    # 对每个坐标进行插值
    resampled = np.zeros((num_points, 3))
    for i in range(3):  # x, y, z
        resampled[:, i] = np.interp(t, distances, points[:, i])

    return resampled

def generate_time_slices(origin_data, dest_data, Tmax, dz):
    """
    构建合法的时空棱柱点集
    Args:
        origin_data: 起点等时圈数据，z=tO（从 O1 出发，往上扩张）
        dest_data: 终点等时圈数据，z=Tmax-tD（从 D1 反向往下收拢）
        Tmax: 最大时间值
        dz: 层间距
    Returns:
        valid_points_dict: 所有合法点的分层点集
        origin_points_dict: 起点体的分层点集（用于构建网格）
        dest_points_dict: 终点体的分层点集（用于构建网格）
    """
    # 1. 构建插值函数（使用原始 tO 和 tD）
    tO_func = create_interpolation_function(origin_data)  # 使用原始 tO
    tD_func = create_interpolation_function(dest_data)    # 使用原始 tD

    # 2. 获取空间范围
    x_range = [
        min([p['x'] for fid in origin_data for p in origin_data[fid]] + [p['x'] for fid in dest_data for p in dest_data[fid]]),
        max([p['x'] for fid in origin_data for p in origin_data[fid]] + [p['x'] for fid in dest_data for p in dest_data[fid]])
    ]
    y_range = [
        min([p['y'] for fid in origin_data for p in origin_data[fid]] + [p['y'] for fid in dest_data for p in dest_data[fid]]),
        max([p['y'] for fid in origin_data for p in origin_data[fid]] + [p['y'] for fid in dest_data for p in dest_data[fid]])
    ]

    # 3. 获取起点和终点
    O1 = min([(p['x'], p['y'], p['z']) for fid in origin_data for p in origin_data[fid]], key=lambda x: x[2])
    D1 = min([(p['x'], p['y'], p['z']) for fid in dest_data for p in dest_data[fid]], key=lambda x: x[2])
    print(f"起点O1: {O1}")
    print(f"终点D1(还未反转，即z值是tD=0): {D1}")

    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # 5. 初始化点集字典 - 只包含起点和终点
    max_layer = int(Tmax // dz)  # 计算最大层号
    print(f"层间距 dz: {dz}")
    print(f"最大层号: {max_layer}")

    # 创建所有可能的层号
    all_layers = list(range(0, max_layer + 1))
    print(f"总层数: {len(all_layers)}")

    valid_points_dict = {
        0: [{"x": O1[0], "y": O1[1], "z": O1[2]}],         # 起点
        max_layer: [{"x": D1[0], "y": D1[1], "z": Tmax}]  # 终点
    }
    origin_points_dict = {0: [{"x": O1[0], "y": O1[1], "z": O1[2]}]}  # 起点体
    dest_points_dict = {0: [{"x": D1[0], "y": D1[1], "z": Tmax}]}   # 终点体

    # 6. 遍历空间网格点
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]

            tO = float(tO_func(x, y))  # 原始 tO 值
            tD = float(tD_func(x, y))  # 原始 tD 值

            # 检查时间约束
            if (tO is not None and tD is not None and 
                (tO + tD) <= Tmax and  # 注意括号
                0 <= tO <= Tmax and 
                0 <= tD <= Tmax):
                # 计算起点体层号（从下往上）
                origin_layer = int(tO // dz)

                # 计算终点体层号（从上往下）
                dest_z = Tmax - tD  # 转换为从上往下的z值
                dest_layer = int((Tmax - dest_z) // dz)  # 从上往下计算层号，从0开始

                # 只处理非起点终点层的点
                if origin_layer != 0:
                    if origin_layer not in origin_points_dict:
                        origin_points_dict[origin_layer] = []
                    origin_points_dict[origin_layer].append({
                        "x": x,
                        "y": y,
                        "z": tO
                    })

                    if origin_layer not in valid_points_dict:
                        valid_points_dict[origin_layer] = []
                    valid_points_dict[origin_layer].append({
                        "x": x,
                        "y": y,
                        "z": tO
                    })

                # 处理终点体点（非终点层）
                if dest_layer != 0:  # 注意这里改为 != 0
                    # 添加到终点体 (x, y, Tmax-tD)
                    if dest_layer not in dest_points_dict:
                        dest_points_dict[dest_layer] = []
                    dest_points_dict[dest_layer].append({
                        "x": x,
                        "y": y,
                        "z": dest_z  # 使用 Tmax-tD
                    })

                    if dest_layer not in valid_points_dict:
                        valid_points_dict[dest_layer] = []
                    valid_points_dict[dest_layer].append({
                        "x": x,
                        "y": y,
                        "z": dest_z
                    })

    return valid_points_dict, origin_points_dict, dest_points_dict

def visualize_all_points(origin_data, dest_data, Tmax):
    """可视化原始点集"""
    origin_points = []
    for fid in origin_data:
        for p in origin_data[fid]:
            origin_points.append([p['x'], p['y'], p['z']])

    # 收集终点数据的点（注意要反转z值）
    dest_points = []
    for fid in dest_data:
        for p in dest_data[fid]:
            dest_points.append([p['x'], p['y'], Tmax - p['z']])  # 反转z值

    # 转换为numpy数组
    origin_points = np.array(origin_points)
    dest_points = np.array(dest_points)

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制起点数据点（灰色）
    ax.scatter(origin_points[:, 0], origin_points[:, 1], origin_points[:, 2],
              c='gray', alpha=0.5, s=2, label='起点体点集')

    # 绘制终点数据点（紫色）
    ax.scatter(dest_points[:, 0], dest_points[:, 1], dest_points[:, 2],
              c='purple', alpha=0.5, s=2, label='终点体点集')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title('原始点集')

    # 设置z轴范围
    ax.set_zlim(0, Tmax)

    # 添加图例
    ax.legend()

    # 显示图形
    plt.show()

def visualize_valid_points(points_dict, Tmax, trip_time_db, title="合法点集"):
    """使用 matplotlib 可视化满足约束的点集"""
    # 将字典中的点转换为数组
    points = np.array([[p["x"], p["y"], p["z"]] for points in points_dict.values() for p in points])

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云，使用z值作为颜色
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=points[:, 2], cmap='viridis',
                        alpha=0.6, s=2)

    # # 添加颜色条
    # plt.colorbar(scatter, label='时间 (Z)')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title(title)

    # 设置z轴范围
    ax.set_zlim(0, trip_time_db)

    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))

    plt.show()

def visualize_mesh(mesh, Tmax):
    """可视化网格"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 获取顶点和面片
    verts = mesh.vertices
    faces = mesh.faces

    # 绘制网格
    surf = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                           triangles=faces,
                           alpha=0.8,
                           cmap='viridis',
                           edgecolor='white',
                           linewidth=0.2)

    # 设置坐标轴范围
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')

    # 设置z轴范围，确保起点在底部，终点在顶部
    ax.set_zlim(0, Tmax)

    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))

    plt.title('时空棱柱')
    plt.show()

def visualize_convex_hull(points_dict, Tmax):
    """可视化整体点集的凸包并返回mesh对象"""
    # 将字典中的点转换为数组
    all_points = np.array([[p["x"], p["y"], p["z"]] for points in points_dict.values() for p in points])

    # 计算凸包
    hull = ConvexHull(all_points)

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始点
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
               c='blue', alpha=0.1, label='原始点')

    # 绘制凸包面
    for simplex in hull.simplices:
        pts = all_points[simplex]
        # 使用plot而不是plot_trisurf来绘制凸包面的边界
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-', alpha=0.3)

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title('时空棱柱凸包')

    # 设置z轴范围
    ax.set_zlim(0, Tmax)

    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))

    plt.show()

    try:
        # 使用hull.points而不是hull.vertices来获取正确的顶点集
        vertices = all_points
        faces = hull.simplices
        
        # 验证面片索引是否在顶点范围内
        if np.max(faces) >= len(vertices):
            print(f"警告：面片索引超出顶点范围，最大索引：{np.max(faces)}，顶点数量：{len(vertices)}")
            return None
            
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    except Exception as e:
        print(f"创建网格时出错: {e}")
        return None

def visualize_dual_convex_hull(origin_data, dest_data, valid_points_dict, Tmax):
    """同时可视化原始点集和合法点集的凸包"""
    # 收集原始点集
    origin_points = []
    for fid in origin_data:
        for p in origin_data[fid]:
            origin_points.append([p['x'], p['y'], p['z']])

    dest_points = []
    for fid in dest_data:
        for p in dest_data[fid]:
            dest_points.append([p['x'], p['y'], Tmax - p['z']])  # 反转z值

    # 收集合法点集
    valid_points = [[p["x"], p["y"], p["z"]] for points in valid_points_dict.values() for p in points]

    # 转换为numpy数组
    origin_points = np.array(origin_points)
    dest_points = np.array(dest_points)
    valid_points = np.array(valid_points)

    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始点集
    ax.scatter(origin_points[:, 0], origin_points[:, 1], origin_points[:, 2],
              c='gray', alpha=0.2, s=2, label='起点原始点集')
    ax.scatter(dest_points[:, 0], dest_points[:, 1], dest_points[:, 2],
              c='purple', alpha=0.2, s=2, label='终点原始点集')

    # 绘制合法点集
    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
              c='green', alpha=0.2, s=2, label='合法点集')

    # 计算并绘制起点凸包
    if len(origin_points) >= 4:  # 至少需要4个点才能生成三维凸包
        try:
            origin_hull = ConvexHull(origin_points)
            # 绘制起点凸包面片
            for simplex in origin_hull.simplices:
                verts = origin_points[simplex]
                # 创建三角面
                poly = Poly3DCollection([verts], alpha=0.3)
                poly.set_color('blue')
                poly.set_edgecolor('lightblue')
                ax.add_collection3d(poly)
            print(f"已生成起点凸包，面片数: {len(origin_hull.simplices)}")
        except Exception as e:
            print(f"生成起点凸包时出错: {str(e)}")
    else:
        print("起点数据点不足，无法生成凸包")

    # 计算并绘制终点凸包
    if len(dest_points) >= 4:
        try:
            dest_hull = ConvexHull(dest_points)
            # 绘制终点凸包面片
            for simplex in dest_hull.simplices:
                verts = dest_points[simplex]
                # 创建三角面
                poly = Poly3DCollection([verts], alpha=0.3)
                poly.set_color('red')
                poly.set_edgecolor('lightcoral')
                ax.add_collection3d(poly)
            print(f"已生成终点凸包，面片数: {len(dest_hull.simplices)}")
        except Exception as e:
            print(f"生成终点凸包时出错: {str(e)}")
    else:
        print("终点数据点不足，无法生成凸包")

    # 计算并绘制合法点集的凸包
    if len(valid_points) >= 4:
        try:
            valid_hull = ConvexHull(valid_points)
            # 绘制合法点集凸包面片
            for simplex in valid_hull.simplices:
                verts = valid_points[simplex]
                # 创建三角面
                poly = Poly3DCollection([verts], alpha=0.15)
                poly.set_color('green')
                poly.set_edgecolor('lightgreen')
                ax.add_collection3d(poly)
            print(f"已生成合法点集凸包，面片数: {len(valid_hull.simplices)}")
        except Exception as e:
            print(f"生成合法点集凸包时出错: {str(e)}")
    else:
        print("合法点集数据点不足，无法生成凸包")

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title('原始点集与合法点集凸包对比')

    # 设置z轴范围
    ax.set_zlim(0, Tmax)

    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))

    # 添加图例
    ax.legend()

    plt.show()

    return None

def generate_loft_surfaces(time_slices, sorted_keys=None, reverse=False, resample_n=30):
    """构建网格
    Args:
        time_slices: 时间切片列表，每个切片是一个点集
        sorted_keys: 可选的层顺序（若为None则自动按z排序）
        reverse: 是否反向连接（终点体需要）
        resample_n: 每层重采样点数，确保结构对齐
    Returns:
        vertices: 顶点列表
        faces: 面片列表
    """
    if len(time_slices) < 2:
        raise ValueError("至少需要两个时间切片才能构建网格")

    if sorted_keys is None:
        sorted_keys = sorted(time_slices.keys(), reverse=reverse)

    # 收集并重采样所有有效层
    layers = []
    for i, k in enumerate(sorted_keys):
        pts = time_slices[k]
        arr = np.array([[p["x"], p["y"], p["z"]] for p in pts])
        if len(arr) < 3:
            continue

        # 只对中间层进行重采样，保持起点和终点层不变
        if i == 0 or i == len(sorted_keys) - 1:  # 起点或终点层
            layers.append(arr)
        else:  # 中间层进行重采样
            resampled = resample_points(arr, resample_n)
            layers.append(resampled)

    if len(layers) < 2:
        raise ValueError("重采样后的有效层数不足，无法构建网格")

    vertices = []
    faces = []
    vertex_count = 0

    # 处理每一层
    for i in range(len(layers)):
        layer = layers[i]
        vertices.extend(layer)

        # 第一层使用Delaunay三角化
        if i == 0:
            tri = Delaunay(layer[:, :2])
            faces.extend(tri.simplices + vertex_count)
        # 其他层构建连接
        elif i < len(layers):
            prev_layer_size = len(layers[i-1])
            curr_layer_size = len(layer)

            # 创建当前层的Delaunay三角化
            tri = Delaunay(layer[:, :2])
            faces.extend(tri.simplices + vertex_count)

            # 连接上一层和当前层
            # 使用最近邻搜索来连接不同大小的层
            prev_points = layers[i-1][:, :2]  # 只使用xy坐标
            curr_points = layer[:, :2]

            # 构建KD树用于最近邻搜索
            tree = cKDTree(prev_points)

            # 对当前层的每个点，找到上一层最近的点
            for j in range(curr_layer_size):
                curr_pt = curr_points[j]
                # 找到最近的3个点
                distances, indices = tree.query(curr_pt, k=min(3, prev_layer_size))

                # 创建连接三角形
                for idx in indices:
                    faces.append([
                        vertex_count + j,  # 当前层的点
                        vertex_count - prev_layer_size + idx,  # 上一层的点
                        vertex_count - prev_layer_size + indices[0]  # 上一层的第一个最近点
                    ])

        vertex_count += len(layer)

    return np.array(vertices), np.array(faces)

def visualize_origin_dest_points(origin_points_dict, dest_points_dict, Tmax, title="起终点合法点集"):
    """分别可视化起点和终点的合法点集"""
    # 将字典中的点转换为数组
    origin_points = np.array([[p["x"], p["y"], p["z"]] for points in origin_points_dict.values() for p in points])
    dest_points = np.array([[p["x"], p["y"], p["z"]] for points in dest_points_dict.values() for p in points])

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制起点体点云（蓝色）
    ax.scatter(origin_points[:, 0], origin_points[:, 1], origin_points[:, 2],
              c='blue', alpha=0.6, s=2, label='起点体点集')

    # 绘制终点体点云（红色）
    ax.scatter(dest_points[:, 0], dest_points[:, 1], dest_points[:, 2],
              c='red', alpha=0.6, s=2, label='终点体点集')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title(title)

    # 设置z轴范围
    # ax.set_zlim(0, Tmax)

    # 添加图例
    ax.legend()

    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))

    plt.show()

def visualize_original_and_valid_points(origin_data, dest_data, valid_points_dict, Tmax):
    """同时可视化原始点集和合法点集"""
    # 收集原始点集
    origin_points = []
    for fid in origin_data:
        for p in origin_data[fid]:
            origin_points.append([p['x'], p['y'], p['z']])

    dest_points = []
    for fid in dest_data:
        for p in dest_data[fid]:
            dest_points.append([p['x'], p['y'], Tmax - p['z']])  # 反转z值

    # 收集合法点集
    valid_points = [[p["x"], p["y"], p["z"]] for points in valid_points_dict.values() for p in points]

    # 转换为numpy数组
    origin_points = np.array(origin_points)
    dest_points = np.array(dest_points)
    valid_points = np.array(valid_points)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制起点原始数据点（浅灰色，小点）
    ax.scatter(origin_points[:, 0], origin_points[:, 1], origin_points[:, 2],
              c='gray', alpha=0.5, s=2, label='起点原始点集')

    # 绘制终点原始数据点（浅紫色，小点）
    ax.scatter(dest_points[:, 0], dest_points[:, 1], dest_points[:, 2],
              c='purple', alpha=0.5, s=2, label='终点原始点集')

    # 绘制合法点集（使用渐变色，较大点）
    scatter = ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                        c=valid_points[:, 2], cmap='viridis',
                        alpha=0.8, s=5, label='合法点集')

    # 添加颜色条
    plt.colorbar(scatter, label='时间 (Z)')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title('原始点集与合法点集对比')

    # 设置z轴范围
    ax.set_zlim(0, Tmax)

    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))

    # 添加图例
    ax.legend()

    plt.show()

def main():
    # PostgreSQL连接字符串
    connection_string = "dbname='prism' user='postgres' host='localhost' port='5432' password='SQL123'"
    # 数据表名称
    table_name1 = "ordered_points_origin_102"
    table_name2 = "ordered_points_dest_102"

    # 读取数据
    origin_data = read_ordered_points_from_postgres(connection_string, table_name1)
    dest_data = read_ordered_points_from_postgres(connection_string, table_name2)

    # 获取最大时间值
    Tmax = max(
        max(p['z'] for p in points)
        for fid, points in origin_data.items()
    )
    print(f"最大时间值: {Tmax}")

    # 1. 可视化原始点集
    print("\n可视化原始点集...")
    visualize_all_points(origin_data, dest_data, Tmax)

    # 2. 采样并筛选合法点集
    print("\n采样并筛选合法点...")
    valid_points_dict, origin_points_dict, dest_points_dict = generate_time_slices(origin_data, dest_data, Tmax, dz=20)

    # 2.1 同时可视化原始点集和合法点集
    print("\n同时可视化原始点集和合法点集...")
    visualize_original_and_valid_points(origin_data, dest_data, valid_points_dict, Tmax)
    
    # 3. 可视化原始点集和合法点集的凸包
    print("\n可视化原始点集和合法点集的凸包...")
    visualize_dual_convex_hull(origin_data, dest_data, valid_points_dict, Tmax)
    
    # # 4. 使用PyMeshLab方法生成并可视化水密网格
    # print("\n使用PyMeshLab方法生成并可视化水密网格...")
    # pymeshlab_mesh = visualize_pymeshlab_mesh(valid_points_dict, Tmax)

    # 5. 可视化所有合法点集
    print("\n可视化满足约束的点集...")
    visualize_valid_points(valid_points_dict, Tmax, trip_time_db = Tmax, title="合法点集（包含起点终点）")

    # 6. 显示合法点集凸包并保存
    print("\n显示合法点集凸包...")
    convex_hull_mesh = visualize_convex_hull(valid_points_dict, Tmax)

    # 7. 创建输出目录并保存网格
    output_dir = r"D:\PycharmProjects\pythonProject1\arc3d可视化\求交集与可拼性判断"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存凸包网格
    if convex_hull_mesh is not None:
        try:
            convex_hull_mesh.export(os.path.join(output_dir, "prism102_convex_hull.stl"))
            print("凸包网格已保存")

            # 保存为GeoJSON格式
            try:
                # 获取mesh的顶点和面
                vertices = convex_hull_mesh.vertices
                faces = convex_hull_mesh.faces
                
                # 创建GeoJSON特征列表
                features = []
                
                # 遍历所有面，创建多边形特征
                for face in faces:
                    # 获取面的顶点坐标
                    face_vertices = vertices[face]
                    
                    # 创建多边形坐标列表（GeoJSON格式）
                    # 注意：GeoJSON使用[longitude, latitude]顺序
                    polygon_coords = [[v[0], v[1], v[2]] for v in face_vertices]
                    # 闭合多边形（首尾相连）
                    polygon_coords.append(polygon_coords[0])
                    
                    # 创建GeoJSON特征
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon_coords]
                        },
                        "properties": {
                            "mesh_type": "convex_hull"
                        }
                    }
                    features.append(feature)
                
                # 创建GeoJSON对象
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": features
                }
                
                # 保存GeoJSON文件
                geojson_file = os.path.join(output_dir, "prism102_convex_hull.geojson")
                print(f"\n保存凸包网格GeoJSON到: {geojson_file}")
                
                import json
                with open(geojson_file, 'w', encoding='utf-8') as f:
                    json.dump(geojson_data, f, ensure_ascii=False, indent=2)
                print("凸包网格GeoJSON保存成功")
                
            except Exception as e:
                print(f"保存凸包网格GeoJSON时出错: {str(e)}")

        except Exception as e:
            print(f"保存凸包网格时出错: {e}")
    else:
        print("无法创建凸包网格")

    # 构建并保存合法点集的网格
    if valid_points_dict:
        try:
            print("\n构建时空棱柱网格...")
            vertices, faces = generate_loft_surfaces(valid_points_dict, reverse=False)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            print("\n显示时空棱柱网格...")
            # visualize_mesh(mesh, Tmax)
            # mesh.export(os.path.join(output_dir, "space_time_prism102.stl"))
            # print("时空棱柱网格已保存")
        except Exception as e:
            print(f"构建时空棱柱网格时出错: {e}")

if __name__ == "__main__":
    main()

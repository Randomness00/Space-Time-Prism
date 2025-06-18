# -*- coding: utf-8 -*-
import numpy as np
import trimesh
import pymeshlab
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import arcpy
import pandas as pd
from typing import Tuple
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKTElement
import psycopg2
import gc

from 3D space_time_prism import generate_time_slices, visualize_convex_hull, \
     visualize_valid_points, visualize_original_and_valid_points
from config import *

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def create_network_dataset(input_gdb, roads_feature_class):
    """创建网络数据集"""
    try:
        # 创建要素数据集
        spatial_reference = arcpy.Describe(roads_feature_class).spatialReference
        feature_dataset = os.path.join(input_gdb, 'network')
        if not arcpy.Exists(feature_dataset):
            arcpy.management.CreateFeatureDataset(input_gdb, 'network', spatial_reference)

        # 设置网络数据集路径
        out_name = "network_ND2"
        network_dataset = os.path.join(feature_dataset, out_name)

        # 如果网络数据集不存在，才创建新的
        if not arcpy.Exists(network_dataset):
            print("网络数据集不存在，开始创建新的网络数据集...")

            # 复制道路要素类到要素数据集中
            roads_network = os.path.join(feature_dataset, 'roads_network')
            if not arcpy.Exists(roads_network):
                arcpy.management.CopyFeatures(roads_feature_class, roads_network)
                print("已复制道路要素类到网络数据集")

            # 创建网络数据集
            result = arcpy.na.CreateNetworkDataset(
                feature_dataset,  # 要素数据集
                out_name,  # 网络数据集名称
                ["roads_network"],  # 参与要素类列表
                "NO_ELEVATION"  # 不使用高程
            )
            print("网络数据集创建完成")

            # 构建网络数据集
            print("开始构建网络...")
            arcpy.na.BuildNetwork(network_dataset)
            print("网络构建完成")
        else:
            print(f"使用已存在的网络数据集: {network_dataset}")

        return network_dataset
    except Exception as e:
        print(f"创建网络数据集时出错: {str(e)}")
        raise

def create_service_area(network_dataset, facilities_layer, output_dir, layer_name, cutoffs):
    """创建服务区分析图层并执行分析"""
    try:
        # 确保输出GDB存在
        output_gdb = os.path.join(output_dir, "output.gdb")
        if not arcpy.Exists(output_gdb):
            print(f"创建输出数据库: {output_gdb}")
            arcpy.management.CreateFileGDB(output_dir, "output")

        # 设置工作空间
        arcpy.env.workspace = output_gdb
        arcpy.env.overwriteOutput = True
        print(f"当前工作空间: {arcpy.env.workspace}")

        # 获取网络数据集的空间参考
        network_sr = arcpy.Describe(network_dataset).spatialReference
        print(f"网络数据集坐标系: {network_sr.name}")

        # 检查设施点
        if not arcpy.Exists(facilities_layer):
            raise Exception(f"设施点要素类不存在: {facilities_layer}")

        # 获取设施点数量
        point_count = int(arcpy.GetCount_management(facilities_layer)[0])
        print(f"\n设施点数量: {point_count}")

        if point_count == 0:
            raise Exception("设施点要素类是空的")

        # 检查设施点的坐标系
        facilities_sr = arcpy.Describe(facilities_layer).spatialReference
        print(f"设施点坐标系: {facilities_sr.name}")

        # 如果坐标系不同，发出警告
        if facilities_sr.name != network_sr.name:
            print(f"警告: 设施点和网络数据集使用不同的坐标系")

        print(f"\n1. 使用已存在的设施点要素类: {facilities_layer}")

        print("\n2. 创建服务区分析图层...")
        result_object = arcpy.na.MakeServiceAreaAnalysisLayer(
            network_dataset,
            layer_name,
            "Driving1",
            "FROM_FACILITIES",
            cutoffs,
            output_type="POLYGONS_AND_LINES",
            geometry_at_overlaps="OVERLAP",
            geometry_at_cutoffs="DISKS",
            polygon_trim_distance=0.5
        )

        # 获取图层对象
        layer_object = result_object.getOutput(0)

        # 添加设施点并设置捕捉参数
        sublayer_names = arcpy.na.GetNAClassNames(layer_object)
        facilities_layer_name = sublayer_names["Facilities"]

        print("3. 添加设施点...")
        print(f"使用设施点要素类: {facilities_layer}")
        print(f"设施点要素类是否存在: {arcpy.Exists(facilities_layer)}")

        # 增加捕捉容差并添加详细的错误处理
        snap_tolerance = "100 Meters"  # 增加捕捉容差
        print(f"使用捕捉容差: {snap_tolerance}")

        try:
            # 添加设施点，使用更大的捕捉容差
            arcpy.na.AddLocations(
                layer_object,
                facilities_layer_name,
                facilities_layer,
                "",
                snap_tolerance,
                "",
                None,
                "MATCH_TO_CLOSEST",
                "APPEND",
                "SNAP",
                snap_tolerance
            )

        except Exception as e:
            print(f"添加设施点时出错: {str(e)}")
            raise

        print("4. 解决服务区分析...")
        arcpy.na.Solve(layer_object)

        # 保存结果
        output_layer_file = os.path.join(output_dir, f"{layer_name}.lyrx")
        layer_object.saveACopy(output_layer_file)
        print(f"5. 结果已保存至: {output_layer_file}")

        return layer_object

    except Exception as e:
        print(f"\n创建服务区分析时出错: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc())
        raise

def process_service_area_results(layer_name, roads_feature_class, output_gdb):
    """处理服务区分析结果"""
    try:
        # 定义输出路径
        output_polygons_shp = os.path.join(os.path.dirname(output_gdb), f"{layer_name}_面.shp")
        output_polygons_gdb = os.path.join(output_gdb, f"{layer_name}_面")

        # 导出服务区面
        print(f"导出服务区面到: {output_polygons_shp}")
        arcpy.conversion.ExportFeatures(f"{layer_name}/面",
                                        output_polygons_shp)

        # 检查导出的面是否存在
        if not arcpy.Exists(output_polygons_shp):
            raise Exception(f"服务区面导出失败，文件不存在: {output_polygons_shp}")
        print(f"服务区面导出成功，开始处理...")

        # 将shapefile复制到地理数据库
        print(f"将shapefile复制到地理数据库: {output_polygons_gdb}")
        arcpy.CopyFeatures_management(output_polygons_shp, output_polygons_gdb)

        # 与路网求交集
        print("创建与路网的交点...")
        intersection_output = os.path.join(output_gdb, "network_intersection_points")
        arcpy.analysis.Intersect([output_polygons_gdb, roads_feature_class],
                                 intersection_output, "ALL", "", "POINT")

        # 多部件转单部件
        print("转换为单点...")
        single_points = os.path.join(output_gdb, "single_points")
        arcpy.MultipartToSinglepart_management(intersection_output, single_points)

        # 添加GroupNo字段
        print("添加GroupNo字段...")
        arcpy.management.AddField(single_points, "GroupNo", "DOUBLE")

        # 清理临时shapefile
        try:
            if arcpy.Exists(output_polygons_shp):
                arcpy.Delete_management(output_polygons_shp)
                print(f"已删除临时shapefile: {output_polygons_shp}")
            # 删除相关文件(.dbf, .shx, etc.)
            for ext in ['.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx']:
                temp_file = output_polygons_shp.replace('.shp', ext)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"已删除临时文件: {temp_file}")
        except Exception as e:
            print(f"清理临时文件时出错: {str(e)}")

        return single_points

    except Exception as e:
        print(f"处理服务区结果时出错: {str(e)}")
        raise

def export_to_postgres(shapefile_path: str, table_name: str, db_connection: str):
    """
    将shapefile导出到PostgreSQL数据库

    Args:
        shapefile_path: shapefile文件路径
        table_name: 要创建的表名
        db_connection: 数据库连接字符串
    """
    try:
        print(f"\n开始导出到PostgreSQL数据库...")
        print(f"读取shapefile: {shapefile_path}")

        # 检查shapefile是否存在
        if not os.path.exists(shapefile_path):
            raise Exception(f"找不到shapefile文件: {shapefile_path}")

        # 读取shapefile
        map_data = gpd.GeoDataFrame.from_file(shapefile_path)

        # 检查几何列是否为空
        if map_data['geometry'].isnull().all():
            raise Exception("几何列中所有数据都为空")

        # 检查几何列的类型
        print(f"几何列类型: {map_data['geometry'].dtype}")

        # 打印几何数据的详细信息
        print(f"数据行数: {len(map_data)}")
        print(f"几何列非空值数量: {map_data['geometry'].count()}")

        # 确保几何数据是有效的
        valid_geometries = map_data['geometry'].apply(lambda x: x is not None and not x.is_empty)
        if not valid_geometries.all():
            print("警告: 存在无效的几何数据，将被过滤")
            map_data = map_data[valid_geometries]

        # 检查坐标系统
        if map_data.crs is None:
            print("警告: 数据没有定义坐标系统，将使用WGS84 (EPSG:4326)")
            map_data.set_crs(epsg=4326, inplace=True)
        else:
            print(f"数据坐标系统: {map_data.crs}")

        # 转换geometry为WKT格式
        try:
            map_data['geometry'] = map_data['geometry'].apply(lambda x: WKTElement(x.wkt, 4326))
        except Exception as e:
            print(f"转换几何数据时出错: {str(e)}")
            raise

        # 创建数据库连接
        engine = create_engine(db_connection)

        # 导出到PostgreSQL
        print(f"导出到表 {table_name}...")
        map_data.to_sql(
            table_name,
            engine,
            if_exists='replace',  # 如果表存在则替换
            index=False,
            dtype={
                'geometry': Geometry(geometry_type='POINT', srid=4326)
            }
        )
        print(f"成功导出到PostgreSQL表 {table_name}")

    except Exception as e:
        print(f"导出到PostgreSQL时出错: {str(e)}")
        raise

def update_point_order(order_index, polygons_layer, points_layer, output_dir, facilities_layer, trip_id=None):
    """更新点的顺序"""
    try:
        print(f"\n开始更新点顺序...")
        print(f"多边形图层: {polygons_layer}")
        print(f"点图层: {points_layer}")

        # 如果trip_id未提供，从多边形图层名称中提取
        if trip_id is None:
            trip_id = polygons_layer.split('_')[-2]  # 获取倒数第二个部分作为行程ID

        # 确保使用正确的多边形图层路径
        if not arcpy.Exists(polygons_layer):
            # 如果提供的路径不存在，尝试在output.gdb中查找
            output_gdb = os.path.join(output_dir, "output.gdb")
            alt_polygons_layer = os.path.join(output_gdb, os.path.basename(polygons_layer))
            if arcpy.Exists(alt_polygons_layer):
                polygons_layer = alt_polygons_layer
                print(f"使用替代多边形图层路径: {polygons_layer}")
            else:
                raise Exception(f"找不到多边形图层: {polygons_layer}")

        def truncate(f, n):
            s = '{}'.format(f)
            i, p, d = s.partition('.')
            return '.'.join([i, (d + '0' * (n + 6))[:n + 6]])

        # 获取输入点图层的字段名
        field_names = [field.name for field in arcpy.ListFields(points_layer)]
        print("输入点图层的字段:", field_names)

        # 查找包含 "FID" 和 "ServiceArea" 的字段
        fid_field = None
        for field in field_names:
            if "FID" in field and "ServiceArea" in field:
                fid_field = field
                break

        if not fid_field:
            raise Exception("找不到合适的FID字段")

        print(f"使用字段: {fid_field}")

        with arcpy.da.SearchCursor(polygons_layer, ["SHAPE@", "OBJECTID"]) as cursor:
            for shp, FID in cursor:
                ext = shp.extent
                UL = arcpy.PointGeometry(ext.upperLeft)
                pLine = shp.boundary()
                aList = []
                dMin = 1e6

                with arcpy.da.SearchCursor(points_layer, ["SHAPE@", fid_field]) as pCur:
                    for line in pCur:
                        pnt = line[0].firstPoint
                        pointFID = line[1]
                        if pointFID == FID:
                            L = pLine.measureOnLine(pnt)
                            d = UL.distanceTo(pnt)
                            if d < dMin:
                                dMin = d
                                lMin = L
                            aList.append([L, pnt])

                aList.sort(key=lambda x: x[0])
                aDict = {truncate(pnt.X, 2) + truncate(pnt.Y, 2): i + 1 for i, (L, pnt) in enumerate(aList)}

                with arcpy.da.UpdateCursor(points_layer, ("SHAPE@", "GroupNo", fid_field)) as pCur:
                    for pnt, no, pointFID in pCur:
                        if pointFID == FID:
                            aKey = truncate(pnt.firstPoint.X, 2) + truncate(pnt.firstPoint.Y, 2)
                            pCur.updateRow((pnt, aDict[aKey], pointFID))

        # 获取FID字段的最大值
        max_fid = 0
        with arcpy.da.SearchCursor(points_layer, [fid_field]) as cursor:
            for row in cursor:
                if row[0] > max_fid:
                    max_fid = row[0]
        print(f"当前{fid_field}最大值: {max_fid}")

        # 添加ToBreak字段（如果不存在）
        field_names = [field.name for field in arcpy.ListFields(points_layer)]
        if "ToBreak" not in field_names:
            print("添加ToBreak字段...")
            arcpy.AddField_management(points_layer, "ToBreak", "DOUBLE")

        # 获取设施点的位置
        if not arcpy.Exists(facilities_layer):
            raise Exception("找不到设施点图层")

        facility_point = None
        with arcpy.da.SearchCursor(facilities_layer, ["SHAPE@"]) as cursor:
            for row in cursor:
                facility_point = row[0]
                break  # 只使用第一个设施点

        if not facility_point:
            raise Exception("未找到设施点")

        # 添加新行
        print(f"添加新行，{fid_field}值为: {max_fid + 1}")
        with arcpy.da.InsertCursor(points_layer, ["SHAPE@", fid_field, "ToBreak", "GroupNo"]) as cursor:
            # 插入新行，使用设施点的位置
            cursor.insertRow([facility_point, max_fid + 1, 0, 1])
            print("新行添加成功")

        # 将排序后的点输出为shp文件
        point_type = "origin" if "Origin" in polygons_layer or "origin" in polygons_layer else "dest"
        output_points_shp = os.path.join(output_dir, f"ordered_points_{point_type}_{trip_id}_{order_index}.shp")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查并删除已存在的文件
        if os.path.exists(output_points_shp):
            print(f"删除已存在的文件: {output_points_shp}")
            arcpy.Delete_management(output_points_shp)
            # 删除相关文件(.dbf, .shx, etc.)
            for ext in ['.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx']:
                temp_file = output_points_shp.replace('.shp', ext)
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        print(f"正在保存排序后的点到: {output_points_shp}")
        arcpy.CopyFeatures_management(points_layer, output_points_shp)
        
        # 验证文件是否成功创建
        if not os.path.exists(output_points_shp):
            raise Exception(f"文件保存失败: {output_points_shp}")
        print(f"已将排序后的点保存至: {output_points_shp}")

        # 导出到PostgreSQL
        db_connection = 'postgresql://postgres:SQL123@localhost:5432/prism'
        table_name = f"ordered_points_{point_type}_{trip_id}_{order_index}"
        export_to_postgres(output_points_shp, table_name, db_connection)
        print(f"已将点数据导出到PostgreSQL表: {table_name}")

        return points_layer

    except Exception as e:
        print(f"更新点顺序时出错: {str(e)}")
        raise

def read_ordered_points_from_postgres(connection_string, table_name, reverse_z=False, scale_factor=10000):
    """从PostgreSQL数据库读取原始数据（保留所有顶点）"""
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

    # 连接到PostgreSQL数据库
    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            for row in cursor.fetchall():
                fid, group_no, x, y, z = row

                # 将Decimal转换为float并对x,y,z进行缩放
                x = float(x) if x is not None else 0.0  # 不再乘以1000
                y = float(y) if y is not None else 0.0  # 不再乘以1000
                z = float(z) if z is not None else 0.0    # Z值不再除以50000

                if fid not in points_dict:
                    points_dict[fid] = []

                points_dict[fid].append({
                    "group_no": group_no,
                    "x": x,
                    "y": y,
                    "z": z
                })

    # Z值反转逻辑（如果需要）
    if reverse_z:
        all_z = [float(p["z"]) for fid in points_dict for p in points_dict[fid]]
        z_max, z_min = max(all_z), min(all_z)
        for fid in points_dict:
            for p in points_dict[fid]:
                p["z"] = float(z_max + z_min - p["z"])

    # 闭合处理
    print("\n开始闭合处理...")
    for fid in points_dict:
        sorted_points = sorted(points_dict[fid], key=lambda x: x["group_no"])
        original_count = len(sorted_points)

        if len(sorted_points) < 2:
            print(f"警告: 层 FID={fid} 为单点层，进行特殊处理")
            # 为单点创建一个小多边形
            base_point = sorted_points[0]
            radius = 0.0001  # 小半径
            num_points = 40 # 增加点数以获得更好的圆形近似
            new_points = []

            # 创建更均匀的圆形点分布
            for i in range(num_points):
                angle = i * (2 * np.pi / num_points)  # 均匀分布在整个圆上
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)

                new_points.append({
                    "group_no": base_point["group_no"] + i + 1,
                    "x": base_point["x"] + dx,
                    "y": base_point["y"] + dy,
                    "z": base_point["z"]
                })

            # 添加偏移点并闭合
            sorted_points.extend(new_points)
            sorted_points.append(sorted_points[0].copy())  # 闭合
            print(f"单点层扩展为 {len(sorted_points)} 个点")

        else:
            # 计算首尾点距离
            first = np.array([sorted_points[0]['x'], sorted_points[0]['y'], sorted_points[0]['z']])
            last = np.array([sorted_points[-1]['x'], sorted_points[-1]['y'], sorted_points[-1]['z']])
            distance = np.linalg.norm(first - last)
            threshold = 1e-6

            print(f"层 FID={fid} 首尾点距离: {distance:.8f} (阈值={threshold})")
            if distance > threshold:
                sorted_points.append(sorted_points[0].copy())
                print(f"添加闭合点，总点数从 {original_count} 变为 {len(sorted_points)}")
            else:
                print("首尾点已闭合，无需添加")

        points_dict[fid] = sorted_points

    return points_dict

def visualize_original_and_valid_points(trip_time_db, origin_data, dest_data, valid_points_dict, Tmax, z_offset):
    """
    同时可视化原始点集和合法点集
    
    Args:
        origin_data: 起点数据
        dest_data: 终点数据
        valid_points_dict: 合法点集
        Tmax: 最大时间
        z_offset: z轴偏移量
        
    Returns:
        tuple: (origin_points, dest_points, valid_points) 处理后的原始点集和合法点集
    """
    # 收集原始点集
    origin_points = []
    for fid in origin_data:
        for p in origin_data[fid]:
            origin_points.append([p['x'], p['y'], p['z'] + z_offset])

    dest_points = []
    for fid in dest_data:
        for p in dest_data[fid]:
            dest_points.append([p['x'], p['y'], Tmax - p['z'] + z_offset])

    # 收集合法点集
    valid_points = [[p["x"], p["y"], p["z"] + z_offset] for points in valid_points_dict.values() for p in points]

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
    # plt.colorbar(scatter, label='时间 (Z)')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title('原始点集与合法点集对比')

    # 设置z轴范围
    zmax = max(np.max(origin_points[:, 2]), np.max(dest_points[:, 2]), np.max(valid_points[:, 2]))
    zmin = min(np.min(origin_points[:, 2]), np.min(dest_points[:, 2]), np.min(valid_points[:, 2]))
    z_range = zmax - zmin
    # ax.set_zlim(zmin - z_range * 0.1, zmax + z_range * 0.1)
    ax.set_zlim(0, trip_time_db)

    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))

    # 添加图例
    ax.legend()

    plt.show()
    
    # 创建凸包可视化窗口
    print("\n生成并可视化三个点集的凸包...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算每个点集的凸包（只有点数足够时才计算）
    if len(origin_points) >= 4:  # 凸包需要至少4个点
        try:
            from scipy.spatial import ConvexHull
            origin_hull = ConvexHull(origin_points)
            # 绘制起点凸包
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
    
    if len(dest_points) >= 4:
        try:
            dest_hull = ConvexHull(dest_points)
            # 绘制终点凸包
            for simplex in dest_hull.simplices:
                verts = dest_points[simplex]
                poly = Poly3DCollection([verts], alpha=0.3)
                poly.set_color('purple')
                poly.set_edgecolor('thistle')
                ax.add_collection3d(poly)
            print(f"已生成终点凸包，面片数: {len(dest_hull.simplices)}")
        except Exception as e:
            print(f"生成终点凸包时出错: {str(e)}")
    else:
        print("终点数据点不足，无法生成凸包")
    
    if len(valid_points) >= 4:
        try:
            valid_hull = ConvexHull(valid_points)
            # 绘制合法点集凸包
            for simplex in valid_hull.simplices:
                verts = valid_points[simplex]
                poly = Poly3DCollection([verts], alpha=0.5)
                poly.set_color('green')
                poly.set_edgecolor('lightgreen')
                ax.add_collection3d(poly)
            print(f"已生成合法点集凸包，面片数: {len(valid_hull.simplices)}")
        except Exception as e:
            print(f"生成合法点集凸包时出错: {str(e)}")
    else:
        print("合法点数据点不足，无法生成凸包")
    
    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('时间')
    ax.set_title('原始点集与合法点集的凸包三维体')
    
    # 设置相同的坐标轴范围（使用所有点集的范围）
    x_min = min(np.min(origin_points[:, 0]), np.min(dest_points[:, 0]), np.min(valid_points[:, 0]))
    x_max = max(np.max(origin_points[:, 0]), np.max(dest_points[:, 0]), np.max(valid_points[:, 0]))
    y_min = min(np.min(origin_points[:, 1]), np.min(dest_points[:, 1]), np.min(valid_points[:, 1]))
    y_max = max(np.max(origin_points[:, 1]), np.max(dest_points[:, 1]), np.max(valid_points[:, 1]))
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    # ax.set_zlim(zmin - z_range * 0.1, zmax + z_range * 0.1)
    ax.set_zlim(0, trip_time_db)

    # 添加图例
    blue_patch = plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.3)
    purple_patch = plt.Rectangle((0, 0), 1, 1, color='purple', alpha=0.3)
    green_patch = plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.5)
    ax.legend([blue_patch, purple_patch, green_patch], ['起点凸包', '终点凸包', '合法点集凸包'])
    
    # 自定义刻度格式化
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: f"{z:.0f}"))
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    plt.show()
    
    # 返回处理后的点集
    return origin_points, dest_points, valid_points

class CarpoolFeasibilityChecker:
    def __init__(self):
        """初始化拼车可行性检查器"""
        self.route_orders = [
            # ["O1", "O2", "D1", "D2"],
            # ["O2", "O1", "D2", "D1"],
            ["O1", "O2", "D2", "D1"],
            ["O2", "O1", "D1", "D2"]
        ]

    def check_all_routes(self, mesh_T1: trimesh.Trimesh, mesh_T2: trimesh.Trimesh,
                         O1: Tuple[float, float], D1: Tuple[float, float],
                         O2: Tuple[float, float], D2: Tuple[float, float],
                         trip_id1: int, trip_id2: int) -> List[Dict]:
        """检查所有可能的拼车路线顺序"""
        # 构建交集棱柱
        TT = self.compute_mesh_intersection(mesh_T1, mesh_T2)
        if TT is None or TT.is_empty:
            print("两个行程的时空棱柱没有交集，无法拼车")
            return []

        results = []

        # 检查每种可能的顺序
        for order in self.route_orders:
            order_index = self._get_current_order_index(order)
            print(f"正在检查第 {order_index} 种顺序: {order}")
            result = self.check_single_order(order_index, order, TT, O1, D1, O2, D2, trip_id1, trip_id2)
            if result["is_mergeable"]:
                results.append(result)

        return results

    def check_single_order(self, order_index, order: List[str], TT: trimesh.Trimesh,
                         O1: Tuple[float, float], D1: Tuple[float, float],
                         O2: Tuple[float, float], D2: Tuple[float, float],
                         trip_id1: int, trip_id2: int) -> Dict:
        """检查单个拼车顺序是否可行"""
        try:
            points = {
                "O1": O1, "D1": D1,
                "O2": O2, "D2": D2
            }
            
            Oa = points[order[0]]
            Ob = points[order[1]]
            Da = points[order[2]]
            Db = points[order[3]]
            
            # 计算所有点的坐标范围
            all_points = np.array([O1, D1, O2, D2])
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)
            
            # 获取TT的坐标范围
            tt_min_x, tt_min_y, tt_min_z = TT.bounds[0]
            tt_max_x, tt_max_y, tt_max_z = TT.bounds[1]
            
            # 计算最终的坐标范围（取最大范围）
            min_x = min(min_x, tt_min_x)
            min_y = min(min_y, tt_min_y)
            max_x = max(max_x, tt_max_x)
            max_y = max(max_y, tt_max_y)
            
            # 添加余量
            x_range = max_x - min_x
            y_range = max_y - min_y
            z_range = tt_max_z - tt_min_z
            
            margin_x = x_range * 0.1
            margin_y = y_range * 0.1
            margin_z = z_range * 0.1
            
            display_x_min = min_x - margin_x
            display_x_max = max_x + margin_x
            display_y_min = min_y - margin_y
            display_y_max = max_y + margin_y
            display_z_min = 0  # z轴从0开始
            display_z_max = tt_max_z + margin_z

            # 设置坐标轴范围的函数
            def set_axis_limits(ax):
                ax.set_xlim(display_x_min, display_x_max)
                ax.set_ylim(display_y_min, display_y_max)
                ax.set_zlim(display_z_min, display_z_max)

            # 1. 可视化T0（两个行程时空棱柱的交集）
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_zlabel('时间')

            # 1. 绘制T0和Ob点
            self.plot_mesh(ax, TT, 'purple', 0.3, 'T0 (交集)')
            self.plot_point(ax, Ob, 0, 'red', 'Ob')
            self.plot_point(ax, Da, 0, 'blue', 'Da')
            
            # 设置坐标轴范围
            set_axis_limits(ax)
            
            plt.title(f'拼车顺序{order}\n1. 两个行程时空棱柱的交集(T0)和Ob、Da点')
            plt.show()

            # 2. 在Ob点处作垂直线，与TT相交
            print(f"\n检查顺序: {' → '.join(order)}")
            print(f"Ob点坐标: {Ob}")
            z_range_ob = self.intersect_vertical_line_with_mesh(Ob, TT, point_name="Ob")
            print('Ob点处垂线与T0相交z值范围：', z_range_ob)
            
            # 如果Ob点处垂线与T0相交z值范围为None，直接返回不可行
            if z_range_ob is None:
                print("Ob点处垂线与T0没有交点，拼车方案不可行")
                return {"is_mergeable": False, "order": order, "reason": "Ob点处垂线与T0没有交点"}

            # 3. 在Da点处作垂直线，与TT相交
            print(f"Da点坐标: {Da}")
            z_range_da = self.intersect_vertical_line_with_mesh(Da, TT, point_name="Da")
            print('Da点处垂线与T0相交z值范围：', z_range_da)
            
            # 如果Da点处垂线与T0相交z值范围为None，直接返回不可行
            if z_range_da is None:
                print("Da点处垂线与T0没有交点，拼车方案不可行")
                return {"is_mergeable": False, "order": order, "reason": "Da点处垂线与T0没有交点"}

            # 4. 构建NT棱柱的z_range
            # 使用Ob点垂线与TT的交点下限和Da点垂线与TT的交点上限
            z_range_nt = [z_range_ob[0], z_range_da[1]]
            print(f'NT棱柱的z_range: {z_range_nt}')

            # 5. 获取Da点所在行程的trip_time和最后到达目的地所属行程的trip_time
            trip_time_da = 0
            trip_time_db = 0
            if Da == D1:
                _, _, trip_time_da = read_taxi_data(trip_id1)
            elif Da == D2:
                _, _, trip_time_da = read_taxi_data(trip_id2)
                
            # 获取最后到达目的地所属行程的trip_time
            if Db == D1:
                _, _, trip_time_db = read_taxi_data(trip_id1)
            elif Db == D2:
                _, _, trip_time_db = read_taxi_data(trip_id2)
                
            print(f"Da点所在行程的trip_time: {trip_time_da}")
            print(f"最后到达目的地所属行程的trip_time: {trip_time_db}")
            
            # 使用Da点所在行程的trip_time构建时空棱柱
            trip_time = trip_time_da

            # 6. 构建以Ob-Da为起终点的时空棱柱
            NT = self.build_spacetime_prism(order_index, trip_time_da, trip_time_db, Ob, Da, z_range=z_range_nt)
            if NT is None:
                return {"is_mergeable": False, "order": order, "reason": "无法构建Ob-Da时空棱柱"}

            # 构建以Oa-Ob为起终点的时空棱柱
            print("构建Oa-Ob网络...")
            # 使用从0开始的z值范围
            oaob_z_range = [0, z_range_ob[1]]
            OaOb = self.build_spacetime_prism(order_index, trip_time_da, trip_time_db, Oa, Ob, z_range=oaob_z_range)
            if OaOb is None:
                return {"is_mergeable": False, "order": order, "reason": "无法构建Oa-Ob时空棱柱"}

            # 构建以Da-Db为起终点的时空棱柱
            print("构建Da-Db网络...")
            # 使用从Da点垂线与TT的交点下限值开始的z值范围
            dadb_z_range = [z_range_da[0], trip_time_db]
            DaDb = self.build_spacetime_prism(order_index, trip_time_da, trip_time_db, Da, Db, z_range=dadb_z_range)
            if DaDb is None:
                return {"is_mergeable": False, "order": order, "reason": "无法构建Da-Db时空棱柱"}

            # 7. 检查NT棱柱是否与TT有交集
            intersection = self.compute_mesh_intersection(NT, TT)
            if intersection is None or intersection.is_empty:
                return {"is_mergeable": False, "order": order, "reason": "NT棱柱与TT没有交集"}

            # 8. 检查Da点的垂线与交集是否有交点
            z_range_final = self.intersect_vertical_line_with_mesh(Da, intersection, point_name="Da与交集")
            if z_range_final is None:
                print("Da点的垂线与NT和T0的交集没有交点，拼车方案不可行")
                return {"is_mergeable": False, "order": order, "reason": "Da点的垂线与NT和TT的交集没有交点"}

            print(f"Da点的垂线与NT和T0的交集的z值范围：{z_range_final}")
            print(f"Da与交集交点1的z值: {z_range_final[0]:.2f}")
            print(f"Da与交集交点2的z值: {z_range_final[1]:.2f}")
            print(f"交点z值差值: {(z_range_final[1] - z_range_final[0]):.2f}")

            # 可视化NT与TT的交集
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')

            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_zlabel('时间')

            # 绘制元素
            self.plot_mesh(ax, TT, 'purple', 0.1, 'T0 (两个行程的交集)')
            # self.plot_mesh(ax, NT, 'green', 0.2, 'ObDa时空棱柱')
            # self.plot_mesh(ax, OaOb, 'blue', 0.2, 'OaOb时空棱柱')
            # self.plot_mesh(ax, DaDb, 'yellow', 0.2, 'DaDb时空棱柱')
            self.plot_mesh(ax, intersection, 'red', 0.6, 'NT与T0的交集')
            # 绘制Ob和Da点
            self.plot_point(ax, Ob, 0, 'red', 'Ob')
            self.plot_point(ax, Da, 0, 'blue', 'Da')
            self.plot_point(ax, Oa, 0, 'green', 'Oa')
            self.plot_point(ax, Db, 0, 'yellow', 'Db')

            # 绘制Da点的垂直线与交集的交点
            if z_range_final is not None:
                ax.scatter([Da[0]], [Da[1]], [z_range_final[0]],
                           color='yellow', s=100, label='Da与交集交点1')
                ax.scatter([Da[0]], [Da[1]], [z_range_final[1]],
                           color='yellow', s=100, label='Da与交集交点2')
                # 绘制垂直线
                ax.plot([Da[0], Da[0]],
                        [Da[1], Da[1]],
                        [0, z_range_final[1]],
                        'y--',
                        linewidth=2,
                        label='Da垂直线')

                # 添加交点z值标签
                ax.text(Da[0], Da[1], z_range_final[0], f'{z_range_final[0]:.2f}',
                        color='black', fontsize=10, ha='right')
                ax.text(Da[0], Da[1], z_range_final[1], f'{z_range_final[1]:.2f}',
                        color='black', fontsize=10, ha='right')

            # 设置坐标轴范围
            set_axis_limits(ax)
            ax.set_zlim(0, trip_time_db)

            plt.title(f'拼车顺序{order}\n4. NT与T0的交集及Da垂线')
            plt.legend()
            plt.show()

            # 9.合并三个棱柱并展示
            print("\n使用trimesh合并三个棱柱...")
            # 合并三个 trimesh 对象
            merged_trimesh = trimesh.util.concatenate([NT, OaOb, DaDb])
            
            # 保存合并后的mesh
            output_dir = os.path.join(VISUALIZATION_DIR, "merged_meshes")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 生成文件名，包含行程ID和顺序信息
            output_file = os.path.join(output_dir, f"merged_mesh_{trip_id1}_{trip_id2}_order{order_index}.stl")
            print(f"\n保存合并后的mesh到: {output_file}")
            merged_trimesh.export(output_file)
            print("mesh保存成功")

            # 保存为GeoJSON格式
            try:
                # 获取mesh的顶点和面
                vertices = merged_trimesh.vertices
                faces = merged_trimesh.faces
                
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
                            "trip_id1": trip_id1,
                            "trip_id2": trip_id2,
                            "order_index": order_index
                        }
                    }
                    features.append(feature)
                
                # 创建GeoJSON对象
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": features
                }
                
                # 保存GeoJSON文件
                geojson_file = os.path.join(output_dir, f"merged_mesh_{trip_id1}_{trip_id2}_order{order_index}.geojson")
                print(f"\n保存GeoJSON到: {geojson_file}")
                
                import json
                with open(geojson_file, 'w', encoding='utf-8') as f:
                    json.dump(geojson_data, f, ensure_ascii=False, indent=2)
                print("GeoJSON保存成功")
                
            except Exception as e:
                print(f"保存GeoJSON时出错: {str(e)}")

            # 可视化合并后的网格
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_zlabel('时间 / s')

            # 绘制合并后的网格
            self.plot_mesh(ax, merged_trimesh, 'purple', 0.5, '合并后的时空棱柱')

            # 绘制关键点
            self.plot_point(ax, Oa, 0, 'green', 'Oa')
            self.plot_point(ax, Ob, 0, 'red', 'Ob')
            self.plot_point(ax, Da, 0, 'blue', 'Da')
            self.plot_point(ax, Db, 0, 'orange', 'Db')

            # 设置坐标轴范围
            set_axis_limits(ax)
            ax.set_zlim(0, trip_time_db)

            plt.title(f'拼车顺序{order}\n5. 合并后的时空棱柱')
            plt.legend()
            plt.show()

            return {
                "is_mergeable": True,
                "order": order,
                "z_range": z_range_final
            }
            
        except Exception as e:
            print(f"检查顺序 {order} 时出错: {str(e)}")
            return {"is_mergeable": False, "order": order, "error": str(e)}

    def _get_current_order_index(self, current_order: List[str]) -> int:
        """
        获取当前顺序在route_orders中的索引号（从1开始）
        
        Args:
            current_order: 当前的路由顺序
            
        Returns:
            当前顺序的索引号（1-4）
        """
        for i, order in enumerate(self.route_orders, 1):
            if order == current_order:
                return i
        return 1  # 默认返回1

    def build_spacetime_prism_from_arcpy(self, O: Tuple[float, float], D: Tuple[float, float],
                                         z_range: List[float],trip_time_da, trip_time_db, order_index) -> Optional[trimesh.Trimesh]:
        """
        使用ArcPy构建时空棱柱，通过时间约束筛选点集

        Args:
            O: 起点坐标 (Ob)
            D: 终点坐标 (Da)
            z_range: z轴范围 [min_z, max_z]
            trip_time: 行程时间（秒）
            current_order: 当前的路由顺序
        """
        try:
            # 检查Network Analyst许可
            if arcpy.CheckExtension("network") == "Available":
                arcpy.CheckOutExtension("network")
            else:
                raise arcpy.ExecuteError("Network Analyst Extension许可不可用。")

            # 设置环境和路径
            base_dir = r"D:\Manhatton_pro2"
            input_gdb = os.path.join(base_dir, "新建文件地理数据库.gdb")
            output_dir = base_dir
            output_gdb = os.path.join(output_dir, "output.gdb")

            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"创建输出目录: {output_dir}")

            # 确保输出地理数据库存在
            if not arcpy.Exists(output_gdb):
                print(f"创建输出地理数据库: {output_gdb}")
                arcpy.management.CreateFileGDB(output_dir, "output")
                if not arcpy.Exists(output_gdb):
                    raise Exception(f"无法创建输出地理数据库: {output_gdb}")

            # 输入数据路径
            roads_feature_class = r"D:\Manhatton_pro2\road\setroaddata.shp"  # shp文件路径

            # 计算cutoffs值
            min_z = min(z_range)
            max_z = max(z_range)
            time_range = max_z - min_z
            cutoffs = [int(np.ceil(time_range * (i + 1) / 12)) for i in range(12)]
            print(f"计算得到的cutoffs值: {cutoffs}")

            # 获取当前顺序的索引号
            print(f"当前是第{order_index}个顺序")

            # 创建临时设施点图层
            temp_facilities_origin = os.path.join(output_gdb, f"temp_facilities_origin_ob_{order_index}")
            temp_facilities_dest = os.path.join(output_gdb, f"temp_facilities_dest_da_{order_index}")

            # 删除已存在的临时设施点要素类
            for temp_fc in [temp_facilities_origin, temp_facilities_dest]:
                if arcpy.Exists(temp_fc):
                    arcpy.Delete_management(temp_fc)
                    print(f"已删除已存在的临时设施点要素类: {temp_fc}")

            # 创建临时设施点要素类（起点）
            print(f"创建起点设施点要素类: {temp_facilities_origin}")
            arcpy.CreateFeatureclass_management(
                output_gdb,
                f"temp_facilities_origin_ob_{order_index}",
                "POINT",
                spatial_reference=arcpy.Describe(roads_feature_class).spatialReference
            )
            print("已创建新的起点设施点要素类")

            # 创建临时设施点要素类（终点）
            print(f"创建终点设施点要素类: {temp_facilities_dest}")
            arcpy.CreateFeatureclass_management(
                output_gdb,
                f"temp_facilities_dest_da_{order_index}",
                "POINT",
                spatial_reference=arcpy.Describe(roads_feature_class).spatialReference
            )
            print("已创建新的终点设施点要素类")

            # 添加Ob点（起点）
            with arcpy.da.InsertCursor(temp_facilities_origin, ["SHAPE@"]) as cursor:
                point = arcpy.Point(O[0], O[1])
                cursor.insertRow([arcpy.PointGeometry(point)])
            print(f"已添加Ob点{O}为设施点")

            # 添加Da点（终点）
            with arcpy.da.InsertCursor(temp_facilities_dest, ["SHAPE@"]) as cursor:
                point = arcpy.Point(D[0], D[1])
                cursor.insertRow([arcpy.PointGeometry(point)])
            print(f"已添加Da点{D}为设施点")

            # 创建网络数据集
            network_dataset = create_network_dataset(input_gdb, roads_feature_class)

            # 创建服务区分析图层 - 使用不同的图层名称和不同的cutoffs
            layer_name_origin = f"ServiceArea_Origin_ob_{order_index}"
            layer_name_dest = f"ServiceArea_Dest_da_{order_index}"

            # 创建和执行服务区分析（起点）
            print("\n开始处理起点服务区分析...")
            layer_object_origin = create_service_area(network_dataset, temp_facilities_origin, output_dir,
                                                      layer_name_origin, cutoffs)
            points_layer_origin = process_service_area_results(layer_name_origin, roads_feature_class, output_gdb)
            update_point_order(order_index,f"{layer_name_origin}_面", points_layer_origin, output_dir, temp_facilities_origin,
                               trip_id="ob")

            # 创建和执行服务区分析（终点）
            print("\n开始处理终点服务区分析...")
            layer_object_dest = create_service_area(network_dataset, temp_facilities_dest, output_dir,
                                                    layer_name_dest, cutoffs)
            points_layer_dest = process_service_area_results(layer_name_dest, roads_feature_class, output_gdb)
            update_point_order(order_index, f"{layer_name_dest}_面", points_layer_dest, output_dir, temp_facilities_dest,
                               trip_id="da")

            for lyr in [layer_name_origin, layer_name_dest]:
                if arcpy.Exists(lyr):
                    try:
                        arcpy.Delete_management(lyr)
                        print(f"清理图层：{lyr}")
                    except Exception as e:
                        print(f"图层清理失败：{lyr} - {str(e)}")

            # 从PostgreSQL读取并处理点数据
            db_connection = 'postgresql://postgres:SQL123@localhost:5432/prism'

            # 1. 从数据库读取起点和终点的点集
            print("\n读取起点数据...")
            points_dict_origin = read_ordered_points_from_postgres(
                db_connection,
                f"ordered_points_origin_ob_{order_index}",
                reverse_z=False
            )

            print("\n读取终点数据...")
            points_dict_dest = read_ordered_points_from_postgres(
                db_connection,
                f"ordered_points_dest_da_{order_index}",
                reverse_z=False
            )

            # 3. 在generate_time_slices之前检查点集
            print(f"起点体点集数量: {sum(len(points) for points in points_dict_origin.values())}")
            print(f"终点体点集数量: {sum(len(points) for points in points_dict_dest.values())}")

            # step 2：使用 z_range 长度作为 Tmax
            Tmax = max_z - min_z
            print(f"\n用于 generate_time_slices 的 Tmax = {Tmax}")

            # step 3：生成时间切片
            print("\n生成时间切片...")
            valid_points_dict, origin_points_dict, dest_points_dict = generate_time_slices(
                points_dict_origin,
                points_dict_dest,
                Tmax,
                dz=10  # 层间距
            )

            print(f"\n调整z值，将所有点的z值加上min_z: {min_z}")

            # 2.1 同时可视化原始点集和合法点集
            print("\n同时可视化原始点集和合法点集...")
            origin_points, dest_points, valid_points = visualize_original_and_valid_points(trip_time_db, points_dict_origin, points_dict_dest, valid_points_dict, Tmax, z_offset = min_z)
            
            plt.close('all')  # 关闭所有图形
            del origin_points, dest_points  # 删除大型数组
            gc.collect()  # 强制垃圾回收

            # 3. 可视化所有合法点集
            print("\n可视化满足约束的点集...")
            # 将NumPy数组转换为字典格式
            valid_points_dict_processed = {0: [{"x": p[0], "y": p[1], "z": p[2]} for p in valid_points]}
            visualize_valid_points(valid_points_dict_processed, Tmax, trip_time_db = trip_time_db, title="合法点集（包含起点终点）")
            

            # 4. 构建网格
            print("\n构建网格...")
            mesh = visualize_convex_hull(valid_points_dict_processed, max_z)
            plt.close('all')  # 关闭所有图形
            gc.collect()  # 强制垃圾回收

            # 4. 可视化结果
            print("\n可视化结果...")
            return mesh

        except Exception as e:
            print(f"构建时空棱柱时出错: {str(e)}")
            import traceback
            print("详细错误信息:")
            print(traceback.format_exc())
            return None
        finally:
            if arcpy.CheckExtension("network") == "Available":
                arcpy.CheckInExtension("network")
            # 清理所有图形
            plt.close('all')
            # 清理变量
            if 'valid_points_dict_processed' in locals():
                del valid_points_dict_processed
            if 'points_dict_origin' in locals():
                del points_dict_origin
            if 'points_dict_dest' in locals():
                del points_dict_dest
            # 强制垃圾回收
            gc.collect()

    def build_spacetime_prism(self, order_index, trip_time_da, trip_time_db, O: Tuple[float, float], D: Tuple[float, float],
                                z_range: List[float]) -> Optional[trimesh.Trimesh]:
            """
            构建时空棱柱

            Args:
                O: 起点坐标
                D: 终点坐标
                z_range: z轴范围
                trip_time: 行程时间（秒）
                current_order: 当前的路由顺序

            Returns:
                时空棱柱网格或None
            """
            try:
                # 计算两点之间的距离
                distance = np.linalg.norm(np.array(O) - np.array(D))

                # 调用新的构建函数
                return self.build_spacetime_prism_from_arcpy(O, D, z_range, trip_time_da, trip_time_db, order_index)

            except Exception as e:
                print(f"构建时空棱柱时出错: {str(e)}")
                return None

    def compute_mesh_intersection(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
        """
        计算两个网格的交集

        Args:
            mesh1, mesh2: 输入网格

        Returns:
            交集网格或None（如果没有交集）
        """
        print("使用 PyMeshLab 布尔操作计算网格交集...")

        try:
            # 确保两个网格都是水密的
            if not mesh1.is_watertight:
                print("网格1不是水密的，尝试修复...")
                mesh1 = self.repair_mesh(mesh1)
                if not mesh1.is_watertight:
                    print("警告：网格1修复后仍不是水密的，使用凸包...")
                    mesh1 = mesh1.convex_hull

            if not mesh2.is_watertight:
                print("网格2不是水密的，尝试修复...")
                mesh2 = self.repair_mesh(mesh2)
                if not mesh2.is_watertight:
                    print("警告：网格2修复后仍不是水密的，使用凸包...")
                    mesh2 = mesh2.convex_hull

            # 预处理网格
            print("预处理网格...")
            # 移除重复面（使用新的推荐方法）
            mesh1.update_faces(mesh1.unique_faces())
            mesh2.update_faces(mesh2.unique_faces())

            # 修复法线
            mesh1.fix_normals()
            mesh2.fix_normals()

            # 移除非流形边
            mesh1.process(validate=True)
            mesh2.process(validate=True)

            # 填充孔洞
            mesh1.fill_holes()
            mesh2.fill_holes()

            # 使用 PyMeshLab 进行布尔操作
            print("执行布尔交集操作...")
            try:
                # 创建 PyMeshLab 网格对象
                ms = pymeshlab.MeshSet()

                # 将 trimesh 网格转换为 PyMeshLab 网格
                # 保存为临时文件（使用 PLY 格式）
                temp_mesh1 = "temp_mesh1.ply"
                temp_mesh2 = "temp_mesh2.ply"
                print("导出网格到临时文件...")
                mesh1.export(temp_mesh1)
                mesh2.export(temp_mesh2)

                # 加载网格到 PyMeshLab
                print("加载网格到 PyMeshLab...")
                ms.load_new_mesh(temp_mesh1)
                ms.load_new_mesh(temp_mesh2)

                # 打印网格信息
                print(f"网格1顶点数: {ms.mesh(0).vertex_number()}, 面片数: {ms.mesh(0).face_number()}")
                print(f"网格2顶点数: {ms.mesh(1).vertex_number()}, 面片数: {ms.mesh(1).face_number()}")

                # 执行布尔交集操作
                print("执行布尔交集操作...")
                try:
                    # 使用 mesh_boolean_intersection 而不是 generate_boolean_intersection
                    ms.mesh_boolean_intersection(first_mesh=0, second_mesh=1)
                except Exception as e:
                    print(f"布尔交集操作失败: {str(e)}")
                    raise

                # 获取结果网格
                print("获取结果网格...")
                result_mesh = ms.current_mesh()

                # 检查结果网格是否有效
                if result_mesh.vertex_number() == 0 or result_mesh.face_number() == 0:
                    raise Exception("布尔操作结果为空")

                # 将结果转换回 trimesh 格式
                temp_result = "temp_result.ply"
                print("保存结果网格...")
                ms.save_current_mesh(temp_result)
                intersection_mesh = trimesh.load(temp_result)

                # 清理临时文件
                print("清理临时文件...")
                os.remove(temp_mesh1)
                os.remove(temp_mesh2)
                os.remove(temp_result)

                if intersection_mesh is not None and not intersection_mesh.is_empty:
                    print("布尔交集操作成功")

                    # 修复交集网格
                    print("修复交集网格...")
                    intersection_mesh = self.repair_mesh(intersection_mesh)

                    # 验证结果
                    if intersection_mesh.is_watertight and intersection_mesh.is_volume:
                        print("成功生成水密交集网格")
                        return intersection_mesh
                    else:
                        print("警告：交集网格不是水密的，尝试进一步修复...")
                        # 尝试使用凸包作为最后的保障
                        if not intersection_mesh.is_watertight:
                            print("使用凸包作为最终保障...")
                            intersection_mesh = intersection_mesh.convex_hull
                            intersection_mesh = self.repair_mesh(intersection_mesh)

                            if intersection_mesh.is_watertight:
                                print("成功生成水密交集网格")
                                return intersection_mesh
                else:
                    print("布尔交集操作未产生有效结果")
                    return None

            except Exception as e:
                print(f"布尔交集操作失败: {str(e)}")
                # 尝试清理临时文件
                try:
                    if os.path.exists(temp_mesh1): os.remove(temp_mesh1)
                    if os.path.exists(temp_mesh2): os.remove(temp_mesh2)
                    if os.path.exists(temp_result): os.remove(temp_result)
                except:
                    pass
                return None

        except Exception as e:
            print(f"计算网格交集时出错: {str(e)}")
            return None

    def repair_mesh(self, mesh):
        """尽量不改变原始数据的网格修复"""
        try:
            # 基本修复（保留原有顶点）
            mesh.process(True)  # 确保顶点顺序
            mesh.process(validate=False)  # 禁用顶点合并

            mesh.update_faces(mesh.unique_faces())  # 移除重复三角面
            mesh.update_faces(mesh.nondegenerate_faces())  # 去除退化三角形面（即面积为 0 或三点共线的面）
            mesh.remove_infinite_values()  # 删除包含 inf、-inf 或 NaN 的顶点坐标

            # 修复法线
            mesh.fix_normals(multibody=True)

            # 检查是否有孔洞
            if not mesh.is_watertight:
                print("检测到孔洞，尝试修复...")

                # 尝试使用 trimesh 内置修复（不合并顶点）
                mesh.fill_holes()

                # 如果修复后仍不是水密的，但又需要布尔运算
                # 可以创建一个副本用于布尔运算，原始网格保持不变
                if not mesh.is_watertight and not mesh.is_volume:
                    print("警告：网格不是水密的，布尔运算可能不准确")

            return mesh
        except Exception as e:
            print(f"网格修复出错: {str(e)}")
            return mesh

    def intersect_vertical_line_with_mesh(self, point_xy: Tuple[float, float],
                                          mesh: trimesh.Trimesh,
                                          point_name: str = "点") -> Optional[List[float]]:
        """计算垂直线与网格的交点"""
        try:
            x_min, y_min, z_min = mesh.bounds[0]
            x_max, y_max, z_max = mesh.bounds[1]

            if not (x_min <= point_xy[0] <= x_max and y_min <= point_xy[1] <= y_max):
                return None

            offsets = [(0, 0),(0.000001, 0), (-0.000001, 0), (0, 0.000001), (0, -0.000001),
                       (0.00001, 0), (-0.00001, 0), (0, 0.00001), (0, -0.00001),
                       (0.0001, 0), (-0.0001, 0), (0, 0.0001), (0, -0.0001)
                       ]

            all_z_values = []

            for dx, dy in offsets:
                ray_origin = np.array([point_xy[0] + dx, point_xy[1] + dy, -1])
                ray_direction = np.array([0, 0, 1])
                locations, _, _ = mesh.ray.intersects_location(
                    ray_origins=[ray_origin],
                    ray_directions=[ray_direction]
                )
                if len(locations) > 0:
                    print(f"  找到{len(locations)}个交点: {[loc[2] for loc in locations]}")
                    all_z_values.extend(locations[:, 2])
                else:
                    print(f"  未找到交点")

            if len(all_z_values) > 0:
                min_z = float(min(all_z_values))
                max_z = float(max(all_z_values))
                print(f"{point_name}垂直线与网格交点z值范围: [{min_z}, {max_z}]")
                return [min_z, max_z]
            
            print(f"{point_name}垂直线未与网格相交")
            return None

        except Exception as e:
            print(f"计算{point_name}垂直线交点时出错: {str(e)}")
            return None

    def any_points_inside_mesh(self, points: List[float], mesh: trimesh.Trimesh) -> bool:
        """判断点是否在网格内部"""
        try:
            tree = trimesh.proximity.ProximityQuery(mesh)
            distances = tree.signed_distance(points)
            return np.any(distances < 0)
        except Exception as e:
            print(f"判断点是否在网格内部时出错: {str(e)}")
            return False

    def get_common_z_range(self, points1: List[float], points2: List[float]) -> Optional[List[float]]:
        """获取两个点列表的交集"""
        try:
            common_z_range = [max(points1[0], points2[0]), min(points1[1], points2[1])]
            if common_z_range[0] < common_z_range[1]:
                return common_z_range
            return None
        except Exception as e:
            print(f"获取交集z值范围时出错: {str(e)}")
            return None

    def visualize_results(self, mesh_T1: trimesh.Trimesh, mesh_T2: trimesh.Trimesh,
                         results: List[Dict], O1: Tuple[float, float], D1: Tuple[float, float],
                         O2: Tuple[float, float], D2: Tuple[float, float]):
        """可视化拼车可行性分析结果"""
        try:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # 计算所有点的坐标范围
            all_points = np.array([O1, D1, O2, D2])
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)
            
            # 获取两个网格的坐标范围
            t1_min_x, t1_min_y, t1_min_z = mesh_T1.bounds[0]
            t1_max_x, t1_max_y, t1_max_z = mesh_T1.bounds[1]
            
            t2_min_x, t2_min_y, t2_min_z = mesh_T2.bounds[0]
            t2_max_x, t2_max_y, t2_max_z = mesh_T2.bounds[1]
            
            # 计算最终的坐标范围（取最大范围）
            min_x = min(min_x, t1_min_x, t2_min_x)
            min_y = min(min_y, t1_min_y, t2_min_y)
            min_z = min(t1_min_z, t2_min_z)
            
            max_x = max(max_x, t1_max_x, t2_max_x)
            max_y = max(max_y, t1_max_y, t2_max_y)
            max_z = max(t1_max_z, t2_max_z)
            
            # 添加余量
            x_range = max_x - min_x
            y_range = max_y - min_y
            z_range = max_z - min_z
            
            margin_x = x_range * 0.1
            margin_y = y_range * 0.1
            margin_z = z_range * 0.1
            
            # 绘制两个网格
            self.plot_mesh(ax, mesh_T1, 'blue', alpha=0.3, label='Trip 1')
            self.plot_mesh(ax, mesh_T2, 'green', alpha=0.3, label='Trip 2')
            
            # 绘制起终点
            self.plot_point(ax, O1, 0, 'red', 'O1')
            self.plot_point(ax, D1, 0, 'red', 'D1')
            self.plot_point(ax, O2, 0, 'blue', 'O2')
            self.plot_point(ax, D2, 0, 'blue', 'D2')

            # 设置坐标轴范围
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
            ax.set_zlim(0, max_z + margin_z)  # z轴从0开始
            
            # 设置标签和视角
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_zlabel('时间 (秒)')
            ax.legend()
            ax.view_init(elev=20, azim=45)
            
            # 格式化z轴刻度
            def format_time(x, p):
                return f'{int(x)}s'
            ax.zaxis.set_major_formatter(FuncFormatter(format_time))
            
            plt.title('时空棱柱可视化')
            plt.show()

        except Exception as e:
            print(f"可视化结果时出错: {str(e)}")

    def plot_point(self, ax, point: Tuple[float, float], z: float, color: str, label: str):
        """绘制点"""
        try:
            ax.scatter(point[0], point[1], z,
                       color=color,
                       label=label,
                       s=200,
                       marker='o',
                       edgecolor='black',
                       linewidth=2)
            ax.text(point[0], point[1], z, label,
                    color='black',
                    fontsize=12,
                    fontweight='bold')
        except Exception as e:
            print(f"绘制点时出错: {str(e)}")


    def plot_mesh(self, ax, mesh: trimesh.Trimesh, color: str, alpha: float, label: str):
        """
        绘制网格
        """
        try:
            # 获取网格的顶点和面
            vertices = mesh.vertices
            faces = mesh.faces

            # 创建Poly3DCollection对象
            poly3d = Poly3DCollection(vertices[faces], alpha=alpha)
            poly3d.set_color(color)
            poly3d.set_edgecolor('k')
            poly3d.set_label(label)

            # 添加到图形
            ax.add_collection3d(poly3d)

        except Exception as e:
            print(f"绘制网格时出错: {str(e)}")

def read_taxi_data(trip_id: int) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """从csv读取指定行程ID的起点和终点坐标以及行程时间"""
    try:
        df = pd.read_csv(POINTS_CSV)
        trip_data = df[df['ID'] == trip_id]

        if len(trip_data) == 0:
            raise ValueError(f"未找到ID为 {trip_id} 的行程")

        origin = (float(trip_data['pickup_longitude'].iloc[0]),
                  float(trip_data['pickup_latitude'].iloc[0]))
        destination = (float(trip_data['dropoff_longitude'].iloc[0]),
                       float(trip_data['dropoff_latitude'].iloc[0]))
        trip_time = float(trip_data['trip_time_in_secs'].iloc[0])

        return origin, destination, trip_time

    except Exception as e:
        print(f"读取行程数据时出错: {str(e)}")
        return None, None, None

def main():
    """主函数"""
    try:
        checker = CarpoolFeasibilityChecker()
        trip_id1 = int(input("请输入第一个行程的tripID: "))
        trip_id2 = int(input("请输入第二个行程的tripID: "))

        O1, D1, _ = read_taxi_data(trip_id1)
        O2, D2, _ = read_taxi_data(trip_id2)

        # 添加坐标输出
        print("\n坐标信息：")
        print(f"第一个行程：")
        print(f"起点 O1: 经度={O1[0]}, 纬度={O1[1]}")
        print(f"终点 D1: 经度={D1[0]}, 纬度={D1[1]}")
        print(f"\n第二个行程：")
        print(f"起点 O2: 经度={O2[0]}, 纬度={O2[1]}")
        print(f"终点 D2: 经度={D2[0]}, 纬度={D2[1]}")

        if O1 is None or O2 is None:
            print("无法获取行程坐标，程序退出")
            return

        mesh_T1 = trimesh.load(os.path.join(VISUALIZATION_DIR, "space_time_prism101_convex_hull.stl"))
        mesh_T2 = trimesh.load(os.path.join(VISUALIZATION_DIR, "space_time_prism102_convex_hull.stl"))

        results = checker.check_all_routes(mesh_T1, mesh_T2, O1, D1, O2, D2, trip_id1, trip_id2)

        print("\n拼车可行性分析结果：")
        if not results:
            print("没有找到可行的拼车方案")
        else:
            for result in results:
                if result["is_mergeable"]:
                    print(f"可拼车路线: {' → '.join(result['order'])}")
                    print(f"时间范围: {result['z_range']}")
                else:
                    print(f"不可拼车路线: {' → '.join(result['order'])}")

        checker.visualize_results(mesh_T1, mesh_T2, results, O1, D1, O2, D2)

    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()


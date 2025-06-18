# -*- coding: utf-8 -*-
import arcpy
import os
import pandas as pd
from typing import Tuple
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKTElement

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
                out_name,        # 网络数据集名称
                ["roads_network"],  # 参与要素类列表
                "NO_ELEVATION"   # 不使用高程
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

def update_point_order(polygons_layer, points_layer, output_dir, facilities_layer, trip_id):
    """更新点的顺序"""
    try:
        print(f"\n开始更新点顺序...")
        print(f"多边形图层: {polygons_layer}")
        print(f"点图层: {points_layer}")

        # 不再需要从文件名解析trip_id，直接使用传入的参数
        print(f"处理行程ID: {trip_id}")
        
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
        point_type = "origin" if "Origin" in polygons_layer else "dest"
        output_points_shp = os.path.join(output_dir, f"ordered_points_{point_type}_{trip_id}.shp")
        arcpy.CopyFeatures_management(points_layer, output_points_shp)
        print(f"已将排序后的点保存至: {output_points_shp}")

        # 导出到PostgreSQL
        db_connection = 'postgresql://postgres:SQL123@localhost:5432/prism'
        table_name = f"ordered_points_{point_type}_{trip_id}"
        export_to_postgres(output_points_shp, table_name, db_connection)

        print("点处理完成")
        return points_layer  # 返回处理后的点图层，而不是shapefile

    except Exception as e:
        print(f"更新点顺序时出错: {str(e)}")
        raise


def create_lines_from_points(points_layer, output_lines):
    """将点转换为线"""
    try:
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

        print(f"使用字段 {fid_field} 创建线")
        transFields = ["ToBreak"]

        arcpy.management.PointsToLine(
            points_layer,
            output_lines,
            fid_field,
            "GroupNo",
            "CLOSE",
            "CONTINUOUS",
            "START",
            transFields
        )
        print("线创建成功")

    except Exception as e:
        print(f"创建线时出错: {str(e)}")
        raise


def read_taxi_data(trip_id: int) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    从taxi2013.csv读取指定行程ID的起点和终点坐标以及行程时间
    
    Args:
        trip_id: 行程ID
        
    Returns:
        (起点坐标, 终点坐标, 行程时间(秒))
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(r'D:\PycharmProjects\pythonProject1\arc3d可视化\求交集与可拼性判断\setpoins.csv')
        
        # 查找指定ID的行程
        trip_data = df[df['ID'] == trip_id]
        
        if len(trip_data) == 0:
            raise ValueError(f"未找到ID为 {trip_id} 的行程")
            
        # 获取起点和终点坐标（使用longitude和latitude）
        origin = (float(trip_data['pickup_longitude'].iloc[0]), 
                 float(trip_data['pickup_latitude'].iloc[0]))
        destination = (float(trip_data['dropoff_longitude'].iloc[0]), 
                      float(trip_data['dropoff_latitude'].iloc[0]))
        
        # 获取行程时间（秒）
        trip_time = float(trip_data['trip_time_in_secs'].iloc[0])
        
        print(f"读取到行程 {trip_id} 的坐标:")
        print(f"起点经度: {origin[0]}, 纬度: {origin[1]}")
        print(f"终点经度: {destination[0]}, 纬度: {destination[1]}")
        print(f"行程时间: {trip_time} 秒")
        
        return origin, destination, trip_time
        
    except Exception as e:
        print(f"读取行程数据时出错: {str(e)}")
        return None


def main():
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

        # 输入数据路径
        roads_feature_class = r"D:\Manhatton_pro2\road\setroaddata.shp"  # shp文件路径
        
        # 获取用户输入的行程ID
        trip_id = int(input("\n请输入要使用的行程ID: "))
        
        # 读取行程的起点和终点坐标以及行程时间
        origin, destination, trip_time = read_taxi_data(trip_id)
        if origin is None:
            raise Exception("无法获取行程坐标")

        # 计算cutoffs值
        max_cutoff = trip_time   # 行程时间加300秒
        cutoffs = [int(max_cutoff * (i + 1) / 12) for i in range(12)]  # 均分为7个值
        print(f"\n计算得到的cutoffs值: {cutoffs}")

        # 创建临时设施点图层
        temp_facilities_origin = os.path.join(output_gdb, "temp_facilities_origin")
        temp_facilities_dest = os.path.join(output_gdb, "temp_facilities_dest")
        
        # 删除已存在的临时设施点要素类
        for temp_fc in [temp_facilities_origin, temp_facilities_dest]:
            if arcpy.Exists(temp_fc):
                arcpy.Delete_management(temp_fc)
                print(f"已删除已存在的临时设施点要素类: {temp_fc}")
        
        # 创建临时设施点要素类（起点）
        arcpy.CreateFeatureclass_management(
            output_gdb,
            "temp_facilities_origin",
            "POINT",
            spatial_reference=arcpy.Describe(roads_feature_class).spatialReference
        )
        print("已创建新的起点设施点要素类")
        
        # 创建临时设施点要素类（终点）
        arcpy.CreateFeatureclass_management(
            output_gdb,
            "temp_facilities_dest",
            "POINT",
            spatial_reference=arcpy.Describe(roads_feature_class).spatialReference
        )
        print("已创建新的终点设施点要素类")
        
        # 添加起点设施点
        with arcpy.da.InsertCursor(temp_facilities_origin, ["SHAPE@", "OBJECTID"]) as cursor:
            point_origin = arcpy.Point(origin[0], origin[1])  # longitude, latitude
            point_geometry_origin = arcpy.PointGeometry(point_origin)
            cursor.insertRow([point_geometry_origin, trip_id])
            print(f"已添加行程 {trip_id} 的起点作为设施点")
        
        # 添加终点设施点
        with arcpy.da.InsertCursor(temp_facilities_dest, ["SHAPE@", "OBJECTID"]) as cursor:
            point_dest = arcpy.Point(destination[0], destination[1])  # longitude, latitude
            point_geometry_dest = arcpy.PointGeometry(point_dest)
            cursor.insertRow([point_geometry_dest, trip_id])
            print(f"已添加行程 {trip_id} 的终点作为设施点")

        # 创建网络数据集
        network_dataset = create_network_dataset(input_gdb, roads_feature_class)

        # 设置服务区分析参数
        layer_name_origin = f"ServiceArea_Origin_{trip_id}"
        layer_name_dest = f"ServiceArea_Dest_{trip_id}"

        # 创建和执行服务区分析（起点）
        print("\n开始处理起点服务区分析...")
        layer_object_origin = create_service_area(network_dataset, temp_facilities_origin, output_dir, layer_name_origin, cutoffs)
        points_layer_origin = process_service_area_results(layer_name_origin, roads_feature_class, output_gdb)
        update_point_order(f"{layer_name_origin}_面", points_layer_origin, output_dir, temp_facilities_origin, trip_id)
        output_lines_origin = os.path.join(output_gdb, f"final_lines_origin_{trip_id}")
        create_lines_from_points(points_layer_origin, output_lines_origin)

        # 创建和执行服务区分析（终点）
        print("\n开始处理终点服务区分析...")
        layer_object_dest = create_service_area(network_dataset, temp_facilities_dest, output_dir, layer_name_dest, cutoffs)
        points_layer_dest = process_service_area_results(layer_name_dest, roads_feature_class, output_gdb)
        update_point_order(f"{layer_name_dest}_面", points_layer_dest, output_dir, temp_facilities_dest, trip_id)
        output_lines_dest = os.path.join(output_gdb, f"final_lines_dest_{trip_id}")
        create_lines_from_points(points_layer_dest, output_lines_dest)

        print("处理完成。")

    except Exception as e:
        import traceback, sys
        tb = sys.exc_info()[2]
        print(f"在第 {tb.tb_lineno} 行发生错误")
        print(str(e))
    finally:
        if arcpy.CheckExtension("network") == "Available":
            arcpy.CheckInExtension("network")

if __name__ == '__main__':
    main()


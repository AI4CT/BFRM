#!/usr/bin/env python3

import numpy as np
from stl import mesh
try:
    from pxr import Usd, UsdGeom, Gf, Vt, Sdf
except ImportError:
    print("错误：未找到 USD 库。请确保已正确安装 USD。")
    print("您可以从以下地址下载并安装 USD：")
    print("https://github.com/PixarAnimationStudios/USD/releases")
    exit(1)
import glob
import os
import re

def natural_sort_key(s):
    # 用于自然排序的辅助函数
    return [float(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+\.?\d*)', s)]

def stl_sequence_to_usd(stl_dir, usd_path, max_frames=10, fps=24, binary_format=False):
    try:
        # 获取所有STL文件并按自然顺序排序
        stl_files = glob.glob(os.path.join(stl_dir, "*.stl"))
        stl_files.sort(key=natural_sort_key)
        
        if not stl_files:
            raise ValueError(f"在目录 {stl_dir} 中没有找到STL文件")
        
        # 限制处理的帧数
        stl_files = stl_files[:max_frames]
        print(f"处理前 {len(stl_files)} 个STL文件")
        
        # 根据格式选择创建USD文件
        if binary_format:
            # 确保文件扩展名为.usdc
            usd_path = os.path.splitext(usd_path)[0] + '.usdc'
            print(f"使用二进制格式导出到: {usd_path}")
        else:
            # 确保文件扩展名为.usda
            usd_path = os.path.splitext(usd_path)[0] + '.usda'
            print(f"使用ASCII格式导出到: {usd_path}")
        stage = Usd.Stage.CreateNew(usd_path)
        
        # 创建默认prim
        default_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
        stage.SetDefaultPrim(default_prim.GetPrim())
        
        # 设置时间范围
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(len(stl_files) - 1)
        stage.SetTimeCodesPerSecond(fps)
        
        # 创建网格prim
        mesh_path = "/World/Mesh"
        mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
        
        # 创建属性
        points_attr = mesh_prim.CreatePointsAttr()
        # normals_attr = mesh_prim.CreateNormalsAttr()
        
        for frame_idx, stl_file in enumerate(stl_files):
            print(f"处理第 {frame_idx + 1}/{len(stl_files)} 帧: {os.path.basename(stl_file)}")
            try:
                stl_mesh = mesh.Mesh.from_file(stl_file)
                
                # 获取当前帧的面片数量
                num_faces = len(stl_mesh.vectors)
                face_vertex_counts = [3] * num_faces
                face_vertex_indices = np.arange(num_faces * 3)
                
                # 设置面片结构
                mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
                mesh_prim.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
                
                # 顶点
                points = stl_mesh.vectors.reshape(-1, 3).astype(np.float32)
                points_attr.Set(Vt.Vec3fArray.FromNumpy(points), frame_idx)
                
                # 法线
                # normals = stl_mesh.normals.astype(np.float32)
                # normals_attr.Set(Vt.Vec3fArray.FromNumpy(normals), frame_idx)
                
                # 设置插值方式
                mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
                mesh_prim.CreateInterpolateBoundaryAttr().Set(UsdGeom.Tokens.edgeAndCorner)
                
                print(f"第 {frame_idx + 1} 帧处理成功，面片数量: {num_faces}")
            except Exception as e:
                print(f"处理文件 {stl_file} 时出错: {str(e)}")
                continue
                
        # 保存USD文件
        stage.Save()
        print(f"转换完成！输出文件：{usd_path}")
        
        # 验证生成的文件
        try:
            test_stage = Usd.Stage.Open(usd_path)
            if test_stage:
                print("USD文件验证成功！")
                # 打印时间范围信息
                print(f"动画时间范围: {test_stage.GetStartTimeCode()} - {test_stage.GetEndTimeCode()}")
                print(f"帧率: {test_stage.GetTimeCodesPerSecond()} fps")
                
                # 验证时间采样
                points_attr = test_stage.GetAttributeAtPath("/World/Mesh.points")
                if points_attr:
                    time_samples = points_attr.GetTimeSamples()
                    print(f"时间采样点: {time_samples}")
            else:
                print("警告：USD文件验证失败！")
        except Exception as e:
            print(f"USD文件验证失败：{str(e)}")
    except Exception as e:
        print(f"发生错误：{str(e)}")
        raise

if __name__ == "__main__":
    # 使用指定的目录
    stl_dir = r"C:\codebase\BFRM\results\yolo11l-obb\4-reconstruction\Bubbly_flow_stl"
    usd_path = os.path.join(os.path.dirname(stl_dir), "bubbly_flow_test")
    # 使用二进制格式导出
    stl_sequence_to_usd(stl_dir, usd_path, max_frames=10, fps=24, binary_format=True)

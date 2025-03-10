import os
import pandas as pd
import numpy as np
import pyvista as pv
import time
import glob

# 定义输入文件夹
input_folder = r"C:\codebase\BFRM\results\yolo11l-obb\3d_reconstructed"
vtk_file = os.path.join(input_folder, "bubble_visualization.vtk")
time_series_vtk = os.path.join(input_folder, "bubbles_time_series.vtm")
stl_folder = os.path.join(input_folder, "stl_files")

# 读取CSV文件作为备份方案
def load_csv_frame(frame_num):
    """加载特定帧的CSV数据"""
    file_path = os.path.join(input_folder, f"frame_{frame_num:04d}.csv")
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None

# 从逐帧VTK文件加载数据
def load_vtk_frame(frame_num):
    """加载特定帧的VTK文件"""
    file_path = os.path.join(stl_folder, f"frame_{frame_num:03d}.vtk")
    if not os.path.exists(file_path):
        print(f"找不到VTK文件: {file_path}")
        return None
    
    try:
        multi_block = pv.read(file_path)
        return multi_block
    except Exception as e:
        print(f"读取VTK文件时出错: {str(e)}")
        return None

# 创建3D可视化 - 使用CSV数据
def visualize_frame_from_csv(frame_num=0):
    """基于CSV数据可视化特定帧的气泡"""
    df = load_csv_frame(frame_num)
    if df is None:
        print(f"无法加载帧 {frame_num}")
        return
    
    # 创建一个PyVista绘图对象
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # 设置坐标系
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    # 添加每个气泡
    for _, bubble in df.iterrows():
        # 提取气泡属性
        x, y, z = bubble['x'], bubble['y'], bubble['z']
        width, height = bubble['width'], bubble['height']
        angle = bubble['angle(°)']
        volume = bubble['volume(mm^3)']
        bubble_id = bubble['bubble_id']
        
        # 确定z方向的尺寸（使用width和height的最小值）
        depth = min(width, height)
        
        # 创建椭球体
        sphere = pv.Sphere(radius=1, center=(0, 0, 0))
        
        # 应用缩放和旋转变换
        base_radius = (width + height) / 4
        transform = np.array([
            [width/(2*base_radius), 0, 0, 0],
            [0, height/(2*base_radius), 0, 0],
            [0, 0, depth/(2*base_radius), 0],
            [0, 0, 0, 1]
        ])
        
        # 应用旋转（仅在XY平面）
        angle_rad = np.radians(angle)
        rotation = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 应用平移
        translation = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        
        # 组合变换
        transform = translation @ rotation @ transform
        
        # 应用变换
        bubble_mesh = sphere.copy()
        bubble_mesh.transform(transform)
        
        # 设置气泡颜色（基于体积）
        color = np.array([0.0, 0.5, 0.8])  # 基本颜色：蓝色
        
        # 添加气泡到场景
        plotter.add_mesh(bubble_mesh, color=color, opacity=0.7)
        
    # 显示场景
    plotter.show(title=f"气泡流场（CSV数据） - 帧 {frame_num}")

# 创建3D可视化 - 使用VTK数据
def visualize_frame_from_vtk(frame_num=0):
    """基于VTK文件可视化特定帧的气泡"""
    multi_block = load_vtk_frame(frame_num)
    if multi_block is None:
        print(f"尝试从CSV加载帧 {frame_num}")
        return visualize_frame_from_csv(frame_num)
    
    # 创建一个PyVista绘图对象
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # 设置坐标系和界限
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    plotter.set_bounds([0, 1280, 0, 800, 0, 1280])  # 设置显示范围
    
    # 添加多块数据到场景
    plotter.add_mesh(multi_block, opacity=0.7, color='dodgerblue')
    
    # 显示场景
    plotter.show(title=f"气泡流场（VTK数据） - 帧 {frame_num}")

# 播放时序动画
def play_animation(start_frame=0, end_frame=None, fps=10):
    """播放气泡流场动画"""
    # 找出所有可用的帧
    if end_frame is None:
        vtk_files = glob.glob(os.path.join(stl_folder, "frame_*.vtk"))
        if not vtk_files:
            print("未找到VTK文件，无法播放动画")
            return
        end_frame = len(vtk_files) - 1
    
    # 创建一个离屏渲染的Plotter对象
    plotter = pv.Plotter(off_screen=False)
    plotter.open_movie(os.path.join(input_folder, "animation.mp4"))
    plotter.set_background('white')
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    plotter.set_view_up([0, 0, 1])  # 设置Z轴朝上
    
    print(f"开始播放动画，从帧{start_frame}到帧{end_frame}，FPS={fps}")
    
    # 对每一帧进行渲染
    for frame_num in range(start_frame, end_frame + 1):
        print(f"渲染帧 {frame_num}/{end_frame}")
        
        # 清除上一帧
        plotter.clear()
        plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
        
        # 加载当前帧
        multi_block = load_vtk_frame(frame_num)
        if multi_block is None:
            continue
        
        # 添加到场景
        plotter.add_mesh(multi_block, opacity=0.7, color='dodgerblue')
        plotter.set_background('white')
        
        # 更新标题
        plotter.add_text(f"帧 {frame_num}/{end_frame}", position="upper_edge", font_size=10)
        
        # 写入帧到视频
        plotter.write_frame()
        
        # 控制播放速度
        time.sleep(1/fps)
    
    # 完成视频制作
    plotter.close()
    print(f"动画播放完成，已保存到 {os.path.join(input_folder, 'animation.mp4')}")

# 直接加载时序VTK文件（如果存在）
def visualize_time_series():
    """可视化时序VTK文件"""
    if not os.path.exists(time_series_vtk):
        print(f"时序VTK文件不存在: {time_series_vtk}")
        return
    
    try:
        print(f"加载时序VTK文件: {time_series_vtk}")
        multi_block = pv.read(time_series_vtk)
        
        # 创建一个PyVista绘图对象
        plotter = pv.Plotter()
        plotter.set_background('white')
        
        # 设置坐标系
        plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
        plotter.set_view_up([0, 0, 1])  # 设置Z轴朝上
        
        # 添加到场景
        plotter.add_mesh(multi_block, opacity=0.7, color='dodgerblue')
        
        # 显示场景
        plotter.show(title="气泡流场 - 时序数据")
        
    except Exception as e:
        print(f"加载时序VTK文件时出错: {str(e)}")

# 主函数
def main():
    # 提示用户选择可视化方式
    print("气泡流场可视化程序")
    print("1. 可视化单帧 (VTK)")
    print("2. 可视化单帧 (CSV)")
    print("3. 播放动画")
    print("4. 加载时序VTK文件")
    
    choice = input("请选择功能 (1-4): ")
    
    if choice == '1':
        frame_num = int(input("请输入要可视化的帧号 (0-449): "))
        visualize_frame_from_vtk(frame_num)
    elif choice == '2':
        frame_num = int(input("请输入要可视化的帧号 (0-449): "))
        visualize_frame_from_csv(frame_num)
    elif choice == '3':
        start_frame = int(input("请输入起始帧 (0-449): "))
        end_frame = int(input("请输入结束帧 (0-449): "))
        fps = int(input("请输入帧率 (1-30): "))
        play_animation(start_frame, end_frame, fps)
    elif choice == '4':
        visualize_time_series()
    else:
        print("无效选择")

if __name__ == "__main__":
    main() 
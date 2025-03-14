import os
import cv2
import numpy as np
import pyvista as pv
from tqdm import tqdm
from glob import glob
import argparse

def create_video_from_images(image_pattern, output_path, fps=30):
    """
    从一系列图像创建视频
    
    参数:
        image_pattern: 图像文件路径模式，例如 "frame_*/3d_visualization.png"
        output_path: 输出视频文件路径
        fps: 视频帧率
    """
    # 获取所有匹配的图像文件
    image_files = sorted(glob(image_pattern))
    if not image_files:
        print(f"未找到匹配的图像文件: {image_pattern}")
        return
    
    # 读取第一张图像以获取尺寸
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 写入每一帧 - 优化tqdm显示
    pattern_name = os.path.basename(image_pattern).split('/')[-1]
    print(f"正在生成视频: {output_path} (从 {pattern_name})")
    for image_file in tqdm(image_files, desc=f"生成{pattern_name}视频", unit="帧", 
                          ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        frame = cv2.imread(image_file)
        if frame is not None:
            video_writer.write(frame)
    
    # 释放视频写入器
    video_writer.release()
    print(f"✓ 视频已保存到: {output_path}")

def get_user_camera_position(stl_file, window_size=[1080, 1080]):
    """
    显示第一个STL文件，让用户手动调整视角，并返回相机位置
    
    参数:
        stl_file: STL文件路径
        window_size: 窗口大小
    
    返回:
        相机位置信息
    """
    print(f"正在加载STL文件: {stl_file}")
    print("请调整到合适的视角，然后关闭窗口以继续...")
    
    # 创建交互式绘图器
    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background('white')
    
    # 调整光照设置 - 降低强度
    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.8, light_type='headlight'))
    plotter.add_light(pv.Light(position=(-1, -1, -1), intensity=0.5))
    plotter.add_light(pv.Light(position=(1, -1, 0.5), intensity=0.3))
    
    # 读取STL文件
    mesh = pv.read(stl_file)
    
    # 添加网格到场景 - 使用深蓝色，优化材质属性
    plotter.add_mesh(mesh, color='steelblue', opacity=0.9, 
                    specular=0.5, specular_power=15,
                    smooth_shading=True, show_edges=False, edge_color='gray',
                    ambient=0.3, diffuse=0.7)
    
    # 启用阴影
    plotter.enable_shadows()
    
    # 添加坐标轴
    plotter.add_axes(interactive=True)
    
    # 添加文本提示
    plotter.add_text("调整到合适的视角后关闭窗口", position='upper_left', font_size=20)
    
    # 显示并等待用户交互
    plotter.show()
    
    # 获取用户设置的相机位置
    camera_position = plotter.camera_position
    print(f"获取到相机位置: {camera_position}")
    print(f"相机方向: {plotter.camera.direction}")
    print(f"相机焦点: {plotter.camera.focal_point}")
    print(f"相机上方向: {plotter.camera.up}")
    return camera_position

def create_3d_model_video_with_custom_view(stl_files, output_path, camera_position=None, rotation=False, rotation_axis='z', rotation_speed=1, fps=30):
    """
    从STL文件创建3D模型视频，使用自定义视角或旋转视角
    
    参数:
        stl_files: STL文件路径列表或模式
        output_path: 输出视频文件路径
        camera_position: 自定义相机位置，如果为None则使用第一个STL文件让用户设置
        rotation: 是否旋转模型
        rotation_axis: 旋转轴，可选 'x', 'y', 'z'
        rotation_speed: 旋转速度（每帧旋转的角度）
        fps: 视频帧率
    """
    # 获取所有STL文件
    if isinstance(stl_files, str):
        stl_files = sorted(glob(stl_files))
    if not stl_files:
        print(f"未找到匹配的STL文件")
        return
    
    # 如果没有提供相机位置，让用户设置
    if camera_position is None:
        camera_position = get_user_camera_position(stl_files[0])
    
    # 创建调试输出目录
    debug_dir = os.path.join(os.path.dirname(output_path), "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"创建调试帧目录: {debug_dir}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (1080, 1080))
    
    # 获取场景中心点（焦点）
    first_mesh = pv.read(stl_files[0])
    center = first_mesh.center
    
    # 提取初始相机位置信息
    initial_position = camera_position[0]
    focus_point = center if rotation else camera_position[1]
    up_direction = camera_position[2]
    
    # 设置视频类型描述
    video_type = "旋转视角" if rotation else "固定视角"
    rotation_info = f"绕{rotation_axis}轴旋转，速度{rotation_speed}度/帧" if rotation else ""
    
    print(f"正在生成3D模型视频({video_type}): {output_path}")
    print(f"初始相机位置: {initial_position}")
    print(f"焦点: {focus_point}")
    print(f"上方向: {up_direction}")
    print(f"{rotation_info}")
    
    # 使用tqdm显示更详细的进度信息
    for i, stl_file in enumerate(tqdm(stl_files, desc=f"生成{video_type}视频", unit="帧", 
                                      ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        # 设置离屏渲染
        plotter = pv.Plotter(off_screen=True, window_size=[1080, 1080])
        # 我们不再使用正交投影，而是使用透视投影，以增强三维感
        # plotter.enable_parallel_projection()
        # 设置背景颜色和光照
        plotter.set_background('white')
        
        # 调整光照设置 - 降低强度
        plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.8, light_type='headlight'))
        plotter.add_light(pv.Light(position=(-1, -1, -1), intensity=0.5))
        plotter.add_light(pv.Light(position=(1, -1, 0.5), intensity=0.3))
        # 提取帧号
        # 从文件名中提取时间戳，例如从'bubbly_flow_0.0001.stl'中提取'0.0001'
        time_stamp = os.path.basename(stl_file).split('_')[-1].split('.stl')[0]
        frame_num = int(float(time_stamp) * 10000)  # 将时间戳转换为帧号
        
        # 读取STL文件
        try:
            mesh = pv.read(stl_file)
        except Exception as e:
            print(f"警告: 无法读取STL文件 {stl_file}: {e}")
            continue
        
        # 添加网格到场景 - 使用深蓝色，优化材质属性
        plotter.add_mesh(mesh, color='steelblue', opacity=0.9, 
                        specular=0.5, specular_power=15,
                        smooth_shading=True, show_edges=False, edge_color='gray',
                        ambient=0.3, diffuse=0.7)

        plotter.enable_shadows()
        
        # 添加坐标轴以便于调试
        plotter.add_axes(interactive=False)
        
        # 设置相机位置
        if rotation:
            angle = frame_num * rotation_speed
            # 计算相机位置
            # 从初始位置开始，绕指定轴旋转
            x, y, z = initial_position
            radius = np.sqrt(x**2 + y**2)  # 在xy平面上的距离
            
            if rotation_axis == 'z':
                new_x = radius * np.cos(np.radians(angle))
                new_y = radius * np.sin(np.radians(angle))
                new_z = z
            elif rotation_axis == 'y':
                new_x = radius * np.cos(np.radians(angle))
                new_z = radius * np.sin(np.radians(angle))
                new_y = y
            elif rotation_axis == 'x':
                new_y = radius * np.cos(np.radians(angle))
                new_z = radius * np.sin(np.radians(angle))
                new_x = x
            
            camera_pos = [(new_x, new_y, new_z), focus_point, up_direction]
            plotter.camera_position = camera_pos
        else:
            # 使用固定视角
            plotter.camera_position = camera_position
            camera_pos = camera_position
        
        # 添加文本信息
        plotter.add_text(f"Frame: {frame_num}\nCamera: {camera_pos[0]}\n3D Visualization of Bubble Flow Field", position='upper_left', font_size=20)
        
        # 渲染场景并保存帧
        frame = plotter.screenshot(None, return_img=True)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        
        # plotter.show()
        # 保存每一帧用于调试
        debug_frame_path = os.path.join(debug_dir, f"frame_{frame_num:04d}.png")
        cv2.imwrite(debug_frame_path, frame)
        plotter.close()
    # 释放资源
    video_writer.release()
    
    print(f"✓ 3D模型视频已保存到: {output_path}")
    print(f"✓ 调试帧已保存到: {debug_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从3D模型(STL文件)生成视频')
    parser.add_argument('--base_path', type=str, default=r"C:\codebase\BFRM\results\yolo11l-obb\visualizations_3D",
                        help='可视化数据的基础路径')
    parser.add_argument('--output_path', type=str, default=r"C:\codebase\BFRM\results\yolo11l-obb\videos",
                        help='输出视频的保存路径')
    parser.add_argument('--fps', type=int, default=30,
                        help='视频帧率')
    parser.add_argument('--rotation_speed', type=int, default=0.8,
                        help='旋转速度（度/帧）')
    args = parser.parse_args()
    
    # 基础路径
    base_path = args.base_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print(f"视频生成工具启动")
    print(f"可视化数据路径: {base_path}")
    print(f"输出视频路径: {output_path}")
    print("=" * 60)
    
    # 1. 从PNG图像创建视频
    image_patterns = {
        "3d_visualization": "frame_*/3d_visualization.png",
        "3d_front_view": "frame_*/3d_front_view.png",
        "3d_side_view": "frame_*/3d_side_view.png",
        "3d_top_view": "frame_*/3d_top_view.png",
        "density_estimation": "frame_*/density_estimation.png"
    }
    
    # 取消注释下面的代码，如果需要从PNG图像创建视频
    # print("\n1. 从PNG图像创建视频...")
    # for name, pattern in image_patterns.items():
    #     full_pattern = os.path.join(base_path, pattern)
    #     output_video = os.path.join(output_path, f"{name}.mp4")
    #     create_video_from_images(full_pattern, output_video)
    
    # 2. 从STL文件创建视频
    print("\n2. 从STL文件创建3D模型视频...")
    stl_pattern = os.path.join(base_path, "bubbly_flow_*.stl")
    # 获取所有STL文件
    all_stl_files = sorted(glob(stl_pattern), key=lambda x: float(os.path.basename(x).split('_')[2].split('.stl')[0]))
    total_frames = len(all_stl_files)
    if total_frames == 0:
        print(f"错误: 未找到任何STL文件 ({stl_pattern})")
        return
    
    # 默认处理全部帧
    print(f"找到 {total_frames} 个STL文件，将处理全部文件")
    print(f"从 {os.path.basename(all_stl_files[0])} 到 {os.path.basename(all_stl_files[-1])}")
    
    # 2.1 使用用户自定义视角创建视频
    print("\n2.1 获取用户自定义视角...")
    first_stl = all_stl_files[0]
    custom_camera = get_user_camera_position(first_stl)
    
    # 使用自定义视角创建固定视角视频
    print("\n2.2 创建固定视角视频...")
    custom_output = os.path.join(output_path, "3d_model_static.mp4")
    create_3d_model_video_with_custom_view(all_stl_files, custom_output, camera_position=custom_camera, rotation=False, fps=args.fps)
    
    # 使用自定义视角作为起点，创建绕z轴旋转的视频
    print("\n2.3 创建旋转视角视频...")
    rotation_output = os.path.join(output_path, "3d_model_rotation.mp4")
    create_3d_model_video_with_custom_view(all_stl_files, rotation_output, 
                                            camera_position=custom_camera, 
                                            rotation=True, 
                                            rotation_axis='z', 
                                            rotation_speed=args.rotation_speed,
                                            fps=args.fps)
    
    print("\n所有视频生成完成！")
    print("=" * 60)
    print(f"固定视角视频: {custom_output}")
    print(f"旋转视角视频: {rotation_output}")
    print("=" * 60)

if __name__ == "__main__":
    main() 
import os
import numpy as np
import pyvista as pv
from pyvista import Plotter
from pathlib import Path
from scipy.spatial import ConvexHull
import logging
import time
import imageio.v2 as imageio  # 修复弃用警告
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import signal
import sys
import matplotlib.pyplot as plt
from collections import Counter
import gc  # 添加垃圾回收模块

# 启动虚拟显示以支持无头渲染
# pv.start_xvfb()

# 全局变量用于控制程序运行
running = True

def signal_handler(signum, frame):
    """处理中断信号"""
    global running
    print("\n正在安全停止程序，请稍候...")
    running = False

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BubbleVisualizer:
    def __init__(self, stl_dir):
        """
        初始化气泡可视化器
        Args:
            stl_dir: STL文件所在的目录路径
        """
        self.stl_dir = Path(stl_dir)
        self.bubbles = {}  # 存储所有气泡的网格数据
        self.properties = {}  # 存储所有气泡的属性
        self.camera_position = None  # 存储第一帧的相机位置
        self.first_frame_extracted = False  # 标记是否已提取第一帧相机参数
        self.volume_range = None  # 存储第一帧的体积范围
        self.bubble_colors = {}  # 存储气泡ID到颜色的映射
        
        # 检查目录是否存在
        if not self.stl_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.stl_dir}")
        
        logging.info(f"初始化可视化器，目标目录: {self.stl_dir}")
        
    def correct_mesh(self, mesh, info):
        """
        根据2D检测信息修正3D网格
        Args:
            mesh: 3D网格
            info: 2D检测信息
        Returns:
            pv.PolyData: 修正后的网格
        """
        # 1. 移动到原点
        center = np.array(mesh.center)
        mesh.translate(-center, inplace=True)
        
        # 2. 计算mesh的x方向长度
        x_len = mesh.bounds[1] - mesh.bounds[0]
        
        # 3. 计算目标水平宽度
        angle_rad = np.deg2rad(info['angle'])
        target_width = abs(info['width'] * np.cos(angle_rad)) + abs(info['height'] * np.sin(angle_rad))
        
        # 4. 计算缩放因子
        scale_factor = target_width / x_len if x_len > 0 else 1.0
        
        # 5. 对整体进行均匀缩放
        mesh.scale(scale_factor, inplace=True)
        
        # 6. 旋转回angle
        mesh.rotate_z(-info['angle'], inplace=True)
        
        # 7. 平移到目标位置
        target_position = np.array([info['x'], info['y'], info['z']])
        mesh.translate(target_position, inplace=True)
        
        return mesh

    def load_frames(self, frame, csv_dir=None):
        """
        加载指定帧的气泡
        Args:
            frame: 要加载的帧号
            csv_dir: CSV文件目录路径
        """
        # 清理之前的数据
        if frame in self.bubbles:
            del self.bubbles[frame]
        if frame in self.properties:
            del self.properties[frame]
        
        # 遍历所有气泡文件夹
        bubble_dirs = list(self.stl_dir.iterdir())
        
        frame_bubbles = {}
        frame_properties = {}
        
        # 读取CSV文件
        if csv_dir:
            try:
                # 从浮点数字符串中提取小数部分
                frame_decimal = float(frame)
                frame_num = f"{int(frame_decimal * 10000):04d}"  # 将0.0001转换为0001
                csv_path = Path(csv_dir) / f"frame_{frame_num}.csv"
                if csv_path.exists():
                    bubble_info = read_bubble_csv(csv_path)
                else:
                    logging.warning(f"未找到CSV文件: {csv_path}")
                    bubble_info = {}
            except ValueError:
                logging.error(f"无法处理帧号: {frame}")
                bubble_info = {}
        else:
            bubble_info = {}
        
        for bubble_dir in bubble_dirs:
            if not bubble_dir.is_dir():
                continue
                
            # 查找对应帧的STL文件
            frame_file = bubble_dir / f"{frame}.stl"
            if not frame_file.exists():
                continue
                
            # 从文件夹名称中提取气泡ID（例如：bubble_0001 -> 1）
            try:
                bubble_id = int(bubble_dir.name.split('_')[-1])
            except ValueError:
                logging.error(f"无法从文件夹名称 {bubble_dir.name} 中提取气泡ID")
                continue
            
            try:
                # 使用pyvista加载STL文件
                mesh = pv.read(str(frame_file))
                
                # 如果有CSV信息，进行修正
                if bubble_id in bubble_info:
                    mesh = self.correct_mesh(mesh, bubble_info[bubble_id])
                
                # 计算气泡属性
                volume = mesh.volume
                surface_area = mesh.area
                
                # 计算纵横比
                bounds = mesh.bounds
                dimensions = np.array([
                    bounds[1] - bounds[0],  # x方向
                    bounds[3] - bounds[2],  # y方向
                    bounds[5] - bounds[4]   # z方向
                ])
                aspect_ratio = np.max(dimensions) / np.min(dimensions)
                
                # 计算圆度
                sphericity = (36 * np.pi * volume**2)**(1/3) / surface_area
                
                # 计算凸度
                hull = ConvexHull(mesh.points)
                convex_volume = hull.volume
                convexity = volume / convex_volume
                
                # 存储属性
                frame_properties[bubble_id] = {
                    'volume': volume,
                    'surface_area': surface_area,
                    'aspect_ratio': aspect_ratio,
                    'sphericity': sphericity,
                    'convexity': convexity
                }
                
                # 将属性添加到网格中
                mesh.point_data['volume'] = np.full(mesh.n_points, volume)
                mesh.point_data['surface_area'] = np.full(mesh.n_points, surface_area)
                mesh.point_data['aspect_ratio'] = np.full(mesh.n_points, aspect_ratio)
                mesh.point_data['sphericity'] = np.full(mesh.n_points, sphericity)
                mesh.point_data['convexity'] = np.full(mesh.n_points, convexity)
                
                # 存储网格
                frame_bubbles[bubble_id] = mesh
                
            except Exception as e:
                logging.error(f"处理气泡 {bubble_id} 时出错: {str(e)}")
        
        if frame_bubbles:
            self.bubbles[frame] = frame_bubbles
            self.properties[frame] = frame_properties
            
            # 如果是第一帧，计算体积范围
            if self.volume_range is None:
                volumes = [props['volume'] for props in frame_properties.values()]
                self.volume_range = (min(volumes), max(volumes))
                logging.info(f"设置体积范围: {self.volume_range}")

    def set_camera_position(self, plotter, mesh, is_first_frame=False):
        """
        设置摄像机位置
        Args:
            plotter: PyVista绘图器
            mesh: 网格数据
            is_first_frame: 是否为第一帧
        """
        if (is_first_frame or self.camera_position is None) and not self.first_frame_extracted:
            # 获取场景边界
            bounds = mesh.bounds
            center = np.array([
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ])
            size = np.array([
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4]
            ])
            max_size = np.max(size)
            camera_position = center + np.array([0, -max_size * 2, 0])
            plotter.camera.position = camera_position
            plotter.camera.focal_point = center
            plotter.camera.up = (0, 0, 1)
            plotter.enable_parallel_projection()
            plotter.camera.zoom(1)
            # 保存所有参数
            self.camera_position = {
                'position': plotter.camera.position,
                'focal_point': plotter.camera.focal_point,
                'up': plotter.camera.up,
                'parallel_projection': True,
                'zoom': 1
            }
            self.first_frame_extracted = True
        else:
            # 直接应用第一帧的相机参数
            cam = self.camera_position
            plotter.camera.position = cam['position']
            plotter.camera.focal_point = cam['focal_point']
            plotter.camera.up = cam['up']
            if cam.get('parallel_projection', False):
                plotter.enable_parallel_projection()
            if 'zoom' in cam:
                try:
                    plotter.camera.zoom(cam['zoom'])
                except Exception:
                    pass

    def get_plotter_and_mesh_args(self, window_size=(2160, 2160), background_color='white', color_by='volume', show_edges=False, frame=None):
        """统一获取绘图器和add_mesh参数"""
        plotter = pv.Plotter(window_size=window_size, off_screen=True)
        plotter.set_background(background_color)
        combined = pv.MultiBlock()
        
        if frame is not None:
            bubbles = self.bubbles[frame]
        else:
            bubbles = self.bubbles
            
        for bubble_id, mesh in bubbles.items():
            combined.append(mesh)
            
        mesh_args = dict(
            show_edges=show_edges,
            show_scalar_bar=False,
            lighting=True,
            opacity=1,
            smooth_shading=True,
            specular=1.0,
            specular_power=100,
            metallic=1.0
        )

        if color_by == 'id':
            # 为每个气泡ID分配一个固定的颜色
            if not self.bubble_colors:
                # 如果是第一次渲染，为所有气泡ID分配颜色
                unique_ids = set()
                for frame_bubbles in self.bubbles.values():
                    unique_ids.update(frame_bubbles.keys())
                
                # 从Spectral颜色映射中采样颜色
                np.random.seed(42)
                cmap = plt.colormaps.get_cmap('Spectral')
                n_colors = len(unique_ids)
                color_indices = np.random.permutation(np.linspace(0, 1, n_colors))
                for i, bubble_id in enumerate(sorted(unique_ids)):
                    color = cmap(color_indices[i])[:3]  # 只取RGB
                    color = tuple(int(255 * c) for c in color)  # 转为0~255
                    self.bubble_colors[bubble_id] = color

            # 为每个网格点设置对应的颜色
            for bubble_id, mesh in bubbles.items():
                if bubble_id not in self.bubble_colors:
                    cmap = plt.colormaps.get_cmap('Spectral')
                    color = cmap(np.random.random())[:3]
                    color = tuple(int(255 * c) for c in color)
                    self.bubble_colors[bubble_id] = color
                color = self.bubble_colors[bubble_id]
                # 保证是uint8格式，shape为(n_points, 3)
                mesh.point_data['color'] = np.tile(np.array(color, dtype=np.uint8), (mesh.n_points, 1))
            
            mesh_args['scalars'] = 'color'
            mesh_args['rgb'] = True
        else:
            mesh_args['scalars'] = color_by
            mesh_args['cmap'] = 'Spectral'
            if color_by == 'volume' and self.volume_range is not None:
                min_vol, max_vol = self.volume_range
                mesh_args['clim'] = (min_vol, max_vol)

        return plotter, combined, mesh_args

    def save_combined_mesh(self, output_dir, frame):
        """
        将指定帧的气泡组合成一个网格并保存为STL文件
        Args:
            output_dir: 输出目录路径
            frame: 要保存的帧号
        """
        if frame not in self.bubbles:
            logging.warning(f"帧 {frame} 没有气泡可以保存")
            return
            
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建组合网格
        combined = pv.MultiBlock()
        for bubble_id, mesh in self.bubbles[frame].items():
            combined.append(mesh)
        
        # 将MultiBlock转换为单一网格，并确保是PolyData类型
        combined_mesh = combined.combine().extract_surface()
        
        # 使用帧号作为文件名
        filename = f"{frame}.stl"
        
        # 完整的输出路径
        output_file = output_path / filename
        
        # 保存STL文件
        combined_mesh.save(str(output_file), binary=True)
        logging.info(f"已将帧 {frame} 的组合网格保存到 {output_file}")
        
    def save_visualization(self, output_dir, frame, color_by='volume', show_edges=False, background_color='white'):
        """
        保存指定帧的可视化结果
        Args:
            output_dir: 输出目录路径
            frame: 要保存的帧号
            color_by: 颜色映射的属性
            show_edges: 是否显示边缘
            background_color: 背景颜色
        """
        if frame not in self.bubbles:
            logging.warning(f"帧 {frame} 没有气泡可以保存")
            return
            
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plotter, combined, mesh_args = self.get_plotter_and_mesh_args(
            window_size=(2160, 2160),
            background_color=background_color,
            color_by=color_by,
            show_edges=show_edges,
            frame=frame
        )
        
        # 如果使用体积作为颜色映射，并且有体积范围，则限制颜色范围
        if color_by == 'volume' and self.volume_range is not None:
            min_vol, max_vol = self.volume_range
            mesh_args['clim'] = (min_vol, max_vol)
        
        plotter.add_mesh(combined, **mesh_args)
        
        # 判断是否为第一帧
        is_first_frame = frame == list(self.bubbles.keys())[0]
        self.set_camera_position(plotter, combined, is_first_frame)
        
        # 使用帧号作为文件名
        filename = f"{frame}.png"
        output_file = output_path / filename
        
        plotter.screenshot(
            filename=str(output_file),
            transparent_background=True
        )
        
        # 确保完全关闭和清理
        plotter.close()
        del plotter
        del combined
        del mesh_args
        
        # 强制垃圾回收
        gc.collect()

    def save_gif(self, output_path, color_by='volume', show_edges=False):
        """
        保存GIF动画
        Args:
            output_path: 输出路径
            color_by: 颜色映射的属性
            show_edges: 是否显示边缘
        """
        # 获取所有帧号
        frame_numbers = sorted(self.bubbles.keys())
        
        # 创建临时目录来存储帧图像
        temp_dir = Path(output_path).parent / "temp_frames"
        temp_dir.mkdir(exist_ok=True)
        
        # 读取所有帧的可视化结果
        images = []
        for frame in frame_numbers:
            # 创建临时文件路径
            temp_file = temp_dir / f"frame_{frame}.png"
            
            # 保存当前帧
            self.save_visualization(
                frame=frame,
                output_dir=str(temp_dir),
                color_by=color_by,
                show_edges=show_edges
            )
            
            # 读取保存的图像
            if temp_file.exists():
                images.append(imageio.imread(str(temp_file)))
        
        if not images:
            logging.error("没有成功保存任何帧的图像")
            return
            
        # 生成GIF，设置帧率为10fps
        imageio.mimsave(output_path, images, fps=10)
        logging.info(f"GIF动图已保存到: {output_path}")
        
        # 清理临时文件
        for file in temp_dir.glob("*.png"):
            file.unlink()
        temp_dir.rmdir()

def read_bubble_csv(csv_path):
    """
    读取气泡的CSV信息
    Args:
        csv_path: CSV文件路径
    Returns:
        dict: 气泡信息字典，键为气泡ID，值为包含位置和尺寸信息的字典
    """
    df = pd.read_csv(csv_path)
    # 假设第一列为bubble_id
    bubble_info = {}
    for _, row in df.iterrows():
        bubble_id = int(row['bubble_id'])
        bubble_info[bubble_id] = {
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'width': row['width'],
            'height': row['height'],
            'angle': row['angle(degree)']
        }
    return bubble_info

def assign_bubble_colors(stl_dir):
    cmap = plt.colormaps.get_cmap('Spectral')
    bubble_dirs = list(Path(stl_dir).iterdir())
    unique_ids = set()
    for bubble_dir in bubble_dirs:
        if bubble_dir.is_dir():
            try:
                bubble_id = int(bubble_dir.name.split('_')[-1])
                unique_ids.add(bubble_id)
            except Exception:
                continue
    np.random.seed(42)
    n_colors = len(unique_ids)
    color_indices = np.random.permutation(np.linspace(0, 1, n_colors))
    bubble_colors = {}
    for i, bubble_id in enumerate(sorted(unique_ids)):
        color = cmap(color_indices[i])[:3]
        color = tuple(int(255 * c) for c in color)
        bubble_colors[bubble_id] = color
    return bubble_colors

def main():
    global running
    # 设置基础路径
    base_path = "C:/codebase/BFRM/results/yolo11l-obb"
    # base_path = "/home/yubd/mount/codebase/BFRM/results/RB_bubble_flow1"
    # base_path = "/home/yubd/mount/codebase/BFRM/results/RB_bubble_flow2"
    
    # 使用基础路径构建其他路径
    stl_dir = f"{base_path}/4-reconstruction/bubble_stl_best"
    csv_dir = f"{base_path}/3-3Dbubble_positioning"
    render_dir = Path(f"{base_path}/4-reconstruction/Bubbly_flow_render")
    gif_path = f"{base_path}/4-reconstruction/bubbly_flow.gif"
    
    visualizer = BubbleVisualizer(stl_dir)
    
    # 检查目录是否存在
    if not Path(stl_dir).exists():
        print(f"错误：STL目录不存在: {stl_dir}")
        return
        
    bubble_dirs = list(Path(stl_dir).iterdir())
    bubble_dirs = [d for d in bubble_dirs if d.is_dir()]
    if not bubble_dirs:
        print(f"错误：STL目录中没有子目录: {stl_dir}")
        return
    
    # 从所有气泡文件夹中收集所有帧号
    print("正在收集所有帧号...")
    all_frames = set()
    for bubble_dir in bubble_dirs:
        stl_files = list(bubble_dir.glob("*.stl"))
        for stl_file in stl_files:
            frame_number = stl_file.stem
            all_frames.add(frame_number)
    
    frame_numbers = sorted(all_frames, key=lambda x: float(x))
    print(f"找到 {len(frame_numbers)} 个帧，第一帧: {frame_numbers[0]}, 最后一帧: {frame_numbers[-1]}")
    
    # 确保输出目录存在
    render_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载第一帧获取体积范围
    first_frame = frame_numbers[0]
    try:
        visualizer.load_frames(first_frame, csv_dir)
        volume_range = visualizer.volume_range
        print(f"体积范围: {volume_range}")
    except Exception as e:
        print(f"加载第一帧时出错: {e}")
        return
    
    # 清空缓存，分配颜色
    visualizer.bubbles.clear()
    visualizer.properties.clear()
    bubble_colors = assign_bubble_colors(stl_dir)
    print(f"为 {len(bubble_colors)} 个气泡分配了颜色")
    
    # 处理所有帧
    completed = 0
    failed = 0
    total = len(frame_numbers)
    pbar = tqdm(total=total, desc=f"已完成: {completed} 失败: {failed} 总数: {total}", ncols=100)
    
    try:
        for idx, frame in enumerate(frame_numbers):
            if not running:
                break
            try:
                visualizer.bubble_colors = bubble_colors
                visualizer.load_frames(frame, csv_dir)
                if frame in visualizer.bubbles:
                    visualizer.save_visualization(
                        frame=frame,
                        output_dir=str(render_dir),
                        color_by='id',
                        show_edges=False
                    )
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"处理帧 {frame} 时出错: {e}")
                failed += 1
            
            # 及时清理内存，加强版
            visualizer.bubbles.clear()
            visualizer.properties.clear()
            
            # 每处理10帧强制进行一次垃圾回收
            if idx % 10 == 9:
                gc.collect()
            
            pbar.set_description(f"已完成: {completed} 失败: {failed} 总数: {total}")
            pbar.update(1)
    except KeyboardInterrupt:
        running = False
        pbar.close()
        print("\n已收到中断信号，程序已安全终止。")
        return
    
    pbar.close()
    if not running:
        print("程序已安全停止")
        return
    
    # 生成GIF
    print("正在生成GIF...")
    png_files = sorted(
        [f for f in render_dir.glob("*.png")],
        key=lambda x: float(x.stem)
    )
    if png_files:
        images = [imageio.imread(str(f)) for f in png_files]
        imageio.mimsave(gif_path, images, fps=10)
        print(f"GIF已保存到: {gif_path}")
    else:
        print("没有找到PNG文件，无法生成GIF")

if __name__ == "__main__":
    main() 
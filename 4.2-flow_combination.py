import os
import pyvista as pv
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt
import imageio
import time
from functools import lru_cache
import datetime

@lru_cache(maxsize=100)
def get_csv_cached(csv_dir, frame_num):
    """读取CSV文件的缓存版本"""
    try:
        csv_path = Path(csv_dir) / f"frame_{int(frame_num):04d}.csv"
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV文件失败 {csv_path}: {str(e)}")
        return pd.DataFrame()  # 返回空DataFrame

def get_bubble_info(csv_dir, frame_num, bubble_id):
    """从指定CSV文件中获取气泡信息，带缓存"""
    try:
        df = get_csv_cached(csv_dir, frame_num)
        bubble_data = df[df['bubble_id'] == bubble_id].iloc[0]
        return {
            'x': bubble_data['x'],
            'y': bubble_data['y'],
            'z': bubble_data['z'],
            'width': bubble_data['width'],
            'height': bubble_data['height'],
            'angle': bubble_data['angle(degree)']
        }
    except Exception as e:
        print(f"[get_bubble_info ERROR] frame={frame_num}, bubble_id={bubble_id}, csv={Path(csv_dir) / f'frame_{int(frame_num):04d}.csv'} 错误: {str(e)}")
        return None

def correct_bubble_mesh(mesh, info):
    """根据2D检测信息修正3D网格"""
    try:
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
    except Exception as e:
        print(f"修正网格时出错: {str(e)}")
        return None

def process_single_bubble(args):
    """处理单个气泡文件"""
    file, folder, csv_dir, bubble_stl_dir = args
    try:
        t0 = time.time()
        # 读取obj文件并确保转换为PolyData类型
        mesh = pv.read(str(file))
        t1 = time.time()
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()  # 转换为PolyData
        # 清理mesh以提高性能
        mesh = mesh.clean(inplace=True)
        t2 = time.time()
        # 获取气泡ID和帧号
        bubble_id = int(folder.name.split('_')[1])
        frame_num = file.stem.split('_')[1].split('.')[0]
        # 获取气泡信息并修正mesh
        info = get_bubble_info(csv_dir, frame_num, bubble_id)
        t3 = time.time()
        if info:
            mesh = correct_bubble_mesh(mesh, info)
            t4 = time.time()
            # 保存修正后的mesh到bubble_stl目录，文件名格式为0.0001.stl
            frame_float = float(frame_num) / 10000
            bubble_stl_path = bubble_stl_dir / folder.name / f"{frame_float:.4f}.stl"
            bubble_stl_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.save(str(bubble_stl_path))
            t5 = time.time()
            print(f"[bubble_id={bubble_id} frame={frame_num}] read: {t1-t0:.2f}s, clean: {t2-t1:.2f}s, get_info: {t3-t2:.2f}s, fix: {t4-t3:.2f}s, save: {t5-t4:.2f}s")
            # 统计每个bubble被处理的帧数
            return frame_num, mesh, bubble_id
        return None
    except Exception as e:
        print(f"处理文件 {file} 时出错: {str(e)}")
        return None

def merge_frame_meshes(meshes):
    """合并一帧中的所有气泡mesh"""
    if not meshes:
        return None
    
    try:
        # 确保所有mesh都是PolyData类型
        poly_meshes = []
        for mesh in meshes:
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            poly_meshes.append(mesh)
        
        # 创建一个新的PolyData来存储合并结果
        merged = pv.PolyData()
        
        # 逐个添加mesh
        for mesh in poly_meshes:
            merged = merged.merge(mesh, inplace=True)
        
        # 最后清理一次合并后的mesh
        merged = merged.clean(inplace=True)
        return merged
    except Exception as e:
        print(f"合并mesh时出错: {str(e)}")
        return None

def assign_bubble_colors(bubble_dirs):
    """为气泡分配固定的颜色"""
    cmap = plt.cm.get_cmap('Spectral')
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

def render_frame(plotter, meshes, bubble_colors, camera_params=None):
    """渲染单帧"""
    combined = pv.MultiBlock()
    
    for bubble_id, mesh in meshes.items():
        # 为每个网格点设置对应的颜色
        color = bubble_colors[bubble_id]
        mesh.point_data['color'] = np.tile(np.array(color, dtype=np.uint8), (mesh.n_points, 1))
        combined.append(mesh)
    
    # 添加网格到绘图器
    plotter.add_mesh(combined, scalars='color', rgb=True, show_edges=False)
    
    # 设置相机参数
    if camera_params:
        plotter.camera.position = camera_params['position']
        plotter.camera.focal_point = camera_params['focal_point']
        plotter.camera.up = camera_params['up']
        if camera_params.get('parallel_projection', False):
            plotter.enable_parallel_projection()
        if 'zoom' in camera_params:
            plotter.camera.zoom(camera_params['zoom'])

def process_obj_files(max_bubbles=2):
    # 创建基于时间戳的主输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = Path("C:/codebase/BFRM/results") / timestamp
    
    # 设置路径 - 统一使用session_dir下的子目录
    source_dir = Path("C:/codebase/BFRM/results/RB_bubble_flow2/4-reconstruction/bubble_obj")  # 源目录不变
    csv_dir = Path("C:/codebase/BFRM/results/RB_bubble_flow2/3-3Dbubble_positioning")  # 源CSV目录不变
    
    # 所有输出目录都放在session_dir下
    flow_dir = session_dir / "stl_frames"  # 标准化为stl_frames名称
    bubble_stl_dir = session_dir / "bubble_stl"
    render_dir = session_dir / "renders"
    
    # 创建输出目录
    session_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)
    bubble_stl_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"创建会话目录: {session_dir}")
    print(f"所有输出将保存到该目录下")
    
    # 统计每个bubble的所有帧号
    print("统计每个气泡的活跃帧...")
    bubble_frames = dict()
    for folder in sorted(source_dir.iterdir()):
        if folder.is_dir():
            obj_files = list(folder.glob("*.2.obj"))
            frame_nums = [int(f.stem.split('_')[1].split('.')[0]) for f in obj_files]
            if frame_nums:
                bubble_frames[folder] = frame_nums

    print(f"共发现 {len(bubble_frames)} 个气泡")

    # 为气泡分配颜色
    bubble_colors = assign_bubble_colors(source_dir.iterdir())

    # 收集所有需要处理的文件
    all_files = []
    for folder in sorted(source_dir.iterdir()):
        if folder.is_dir():
            obj_files = list(folder.glob('*.2.obj'))
            for obj_file in obj_files:
                # 提取frame_num
                try:
                    frame_num = obj_file.stem.split('_')[1].split('.')[0]
                except Exception:
                    continue
                all_files.append((obj_file, folder, str(csv_dir), bubble_stl_dir))
    print(f"将处理 {len(all_files)} 个气泡帧文件")
    
    # 统计每个bubble被处理的帧数
    bubble_frame_count = defaultdict(int)
    
    # 使用进程池并行处理文件
    frame_meshes = defaultdict(dict)
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"使用 {n_cores} 个进程并行处理...")
    for_result_bubble_count = defaultdict(int)
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_file = {executor.submit(process_single_bubble, args): args for args in all_files}
        for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="处理文件"):
            result = future.result()
            if result:
                frame_num, mesh, bubble_id = result
                bubble_frame_count[bubble_id] += 1
                for_result_bubble_count[bubble_id] += 1
                frame_meshes[frame_num][bubble_id] = mesh

    # 输出每个bubble被处理的帧数统计
    print("\n每个气泡被处理的帧数统计：")
    for bubble_id in sorted(bubble_frame_count.keys()):
        print(f"bubble_id={bubble_id}: {bubble_frame_count[bubble_id]} 帧")
    
    # 设置第一帧的相机参数
    first_frame = list(frame_meshes.keys())[0]
    first_meshes = frame_meshes[first_frame]
    plotter = pv.Plotter(window_size=(2160, 2160), off_screen=True)
    plotter.set_background('white')
    render_frame(plotter, first_meshes, bubble_colors)
    
    # 获取相机参数
    camera_params = {
        'position': plotter.camera.position,
        'focal_point': plotter.camera.focal_point,
        'up': plotter.camera.up,
        'parallel_projection': True,
        'zoom': 1
    }
    plotter.close()
    
    # 合并每一帧的气泡并渲染
    print("\n合并和渲染每一帧的气泡...")
    for frame_num, meshes in tqdm(frame_meshes.items(), desc="处理帧"):
        # 合并mesh
        merged_mesh = merge_frame_meshes(list(meshes.values()))
        if merged_mesh:
            frame_float = float(frame_num) / 10000
            flow_file = flow_dir / f"{frame_float:.4f}.stl"
            merged_mesh.save(str(flow_file))
        
        # 渲染当前帧
        plotter = pv.Plotter(window_size=(2160, 2160), off_screen=True)
        plotter.set_background('white')
        render_frame(plotter, meshes, bubble_colors, camera_params)
        
        # 保存渲染结果
        render_file = render_dir / f"{frame_float:.4f}.png"
        plotter.screenshot(str(render_file), transparent_background=True)
        plotter.close()
    
    # 生成GIF
    print("\n生成GIF动画...")
    png_files = sorted(
        [f for f in render_dir.glob("*.png")],
        key=lambda x: float(x.stem)
    )
    if png_files:
        images = [imageio.imread(str(f)) for f in png_files]
        gif_path = session_dir / "bubbly_flow.gif"
        imageio.mimsave(str(gif_path), images, fps=10)
        print(f"GIF已保存到: {gif_path}")
    
    # 输出汇总信息
    print("\n处理完成！所有输出已保存到会话目录:")
    print(f"- 会话目录: {session_dir}")
    print(f"- 气泡STL文件: {bubble_stl_dir}")
    print(f"- 合并STL文件: {flow_dir}")
    print(f"- 渲染图像: {render_dir}")
    print(f"- 动画GIF: {session_dir / 'bubbly_flow.gif'}")

if __name__ == "__main__":
    process_obj_files(max_bubbles=2000)

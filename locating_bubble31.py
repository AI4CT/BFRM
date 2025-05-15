import os
import numpy as np
import pandas as pd
import pyvista as pl
from tqdm import tqdm
import matplotlib.cm as cm
from scipy import stats
import random
import matplotlib.pyplot as plt

# 设置matplotlib支持英文显示，避免中文字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义常量
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 800
DEPTH_RANGE = 1280

# 定义气泡半径计算函数
def calc_bubble_radius(row):
    # 使用宽度和高度的平均值的一半作为气泡半径
    return (row['width'] + row['height']) / 4

def calc_bubble_ellipsoid(row):
    """计算气泡的椭球体参数 - 简化版本"""
    # 获取气泡在x-z平面的宽度、高度和角度
    width = row['width']
    height = row['height']
    angle_deg = -row['angle(degree)']
    
    # 将角度转换为弧度
    angle_rad = np.radians(angle_deg)
    
    # 直接设置三个轴的长度
    # a: x轴方向半径
    # b: z轴方向半径
    # c: y轴方向半径（深度）
    a = width / 2  # x轴半径
    b = height / 2  # z轴半径
    c = max(width, height) / 2  # y轴半径（深度）
    
    # 返回椭球体的三个半轴长度和旋转角度
    return {
        'a': a,  # x轴半径
        'b': b,  # z轴半径
        'c': c,  # y轴半径（深度）
        'angle_rad': angle_rad,  # x-z平面上的旋转角度（弧度）
        'angle_deg': angle_deg   # x-z平面上的旋转角度（度）
    }

def read_bubble_csv(csv_path):
    """读取气泡CSV文件，处理可能的编码问题"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin1')
    
    # 确保列名正确
    expected_columns = [
        'bubble_id', 'x', 'y', 'width', 'height', 
        'angle(degree)', 'speed(m/s)', 'volume(mm^3)', 
        'type', 'confidence'
    ]
    
    # 检查必要的列是否存在
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: CSV文件缺少以下必要列: {', '.join(missing_columns)}")
        
        # 为缺失列添加默认值
        for col in missing_columns:
            if col == 'angle(degree)':
                df[col] = 0.0
            elif col == 'type':
                df[col] = 'single'
            elif col == 'confidence':
                df[col] = 1.0
            else:
                df[col] = 0.0
    
    # 确保bubble_id是整数，其他数值列是浮点数，type是字符串
    if 'bubble_id' in df.columns:
        df['bubble_id'] = df['bubble_id'].astype(int)
    
    numeric_columns = [
        'x', 'y', 'width', 'height', 'angle(degree)', 
        'speed(m/s)', 'volume(mm^3)', 'confidence'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    if 'type' in df.columns:
        df['type'] = df['type'].astype(str)
    
    return df

def convert_coordinates(df):
    """转换坐标系并计算气泡参数"""
    # 复制数据框以避免修改原始数据
    df_3d = df.copy()
    
    # 将y轴翻转，使之向上为正（图像坐标系y轴向下为正）
    # 原来的y变为z，保持原始的x不变
    df_3d['z'] = IMAGE_HEIGHT - df_3d['y']
    
    # 计算每个气泡的椭球体参数
    ellipsoid_params = df_3d.apply(calc_bubble_ellipsoid, axis=1)
    
    # 添加椭球体参数到数据框
    df_3d['a'] = ellipsoid_params.apply(lambda x: x['a'])
    df_3d['b'] = ellipsoid_params.apply(lambda x: x['b'])
    df_3d['c'] = ellipsoid_params.apply(lambda x: x['c'])
    df_3d['angle_rad'] = ellipsoid_params.apply(lambda x: x['angle_rad'])
    
    # 保留半径用于兼容性（使用两个主轴的平均值）
    df_3d['radius'] = df_3d.apply(lambda row: (row['a'] + row['b']) / 2, axis=1)
    
    # 创建一个列用于存储y坐标（深度）
    # 初始值设为0，将在assign_y_coordinates中分配实际值
    df_3d['y'] = 0
    
    return df_3d

def assign_y_coordinates(df_3d):
    """为气泡分配y坐标（深度）
    
    在三维坐标系中:
    - x: 表示图像宽度 (1280像素)
    - y: 表示图像深度 (需要分配，符合与x相同的高斯分布)
    - z: 表示图像高度 (800像素)
    """
    # 按类型分组
    df_single = df_3d[df_3d['type'] == 'single'].copy()
    df_overlap = df_3d[df_3d['type'] == 'overlap'].copy()
    
    # 计算x坐标的统计特性
    x_values = df_3d['x'].values
    # 拟合x坐标的正态分布
    mu_x, std_x = stats.norm.fit(x_values)
    print(f"X coordinate distribution fitting result: mean={mu_x:.2f}, standard deviation={std_x:.2f}")
    
    # 确保标准差不会太小，以保证充分的空间分布
    std_x = max(std_x, IMAGE_WIDTH / 10)
    
    # 为single气泡分配y坐标 - 使用与x相同的高斯分布
    # y坐标表示深度，范围为0到DEPTH_RANGE
    if not df_single.empty:
        print("为single气泡分配y坐标(深度)...")
        # 确保y坐标在合理范围内
        y_coords = []
        
        # 按照体积大小排序，大气泡优先
        single_gas = df_single.sort_values('volume(mm^3)', ascending=False)
        
        # 已分配气泡的3D位置
        assigned_bubbles = []
        
        # 为每个单个气泡分配y坐标
        for idx, row in single_gas.iterrows():
            # 创建包含所有必要参数的字典
            current_bubble = {
                'bubble_id': row['bubble_id'],
                'x': row['x'],
                'z': row['z'],
                'radius': row['radius'],
                'volume(mm^3)': row['volume(mm^3)'],
                'type': row['type']
            }
            
            # 添加椭球体参数
            if 'a' in row and 'b' in row and 'c' in row and 'angle_rad' in row:
                current_bubble['a'] = row['a']
                current_bubble['b'] = row['b']
                current_bubble['c'] = row['c']
                current_bubble['angle_rad'] = row['angle_rad']
            
            # 从高斯分布采样，并考虑避免与已分配气泡重叠
            if len(assigned_bubbles) > 0:
                y_coord = find_optimal_y_with_gaussian(
                    current_bubble, 
                    assigned_bubbles, 
                    mu_x, 
                    std_x
                )
            else:
                # 对第一个气泡，直接从高斯分布采样
                y_coord = np.random.normal(mu_x, std_x)
                y_coord = max(0, min(y_coord, DEPTH_RANGE))
            
            # 更新气泡的y坐标
            df_single.at[idx, 'y'] = y_coord
            
            # 添加到已分配列表
            current_bubble['y'] = y_coord
            assigned_bubbles.append(current_bubble)
    
    # 为overlap气泡分配y坐标
    if not df_overlap.empty:
        print("为overlap气泡分配y坐标(深度)...")
        for idx, row in df_overlap.iterrows():
            current_bubble = {
                'bubble_id': row['bubble_id'],
                'x': row['x'],
                'z': row['z'],
                'radius': row['radius'],
                'volume(mm^3)': row['volume(mm^3)'],
                'type': row['type']
            }
            
            # 添加椭球体参数
            if 'a' in row and 'b' in row and 'c' in row and 'angle_rad' in row:
                current_bubble['a'] = row['a']
                current_bubble['b'] = row['b']
                current_bubble['c'] = row['c']
                current_bubble['angle_rad'] = row['angle_rad']
            
            # 尝试为当前气泡找到合适的y坐标
            # 使用与x相同的高斯分布，但尽量避免与已分配气泡重叠
            y_coord = find_optimal_y_with_gaussian(
                current_bubble, 
                assigned_bubbles, 
                mu_x, 
                std_x
            )
            
            # 更新气泡的y坐标
            df_overlap.at[idx, 'y'] = y_coord
            
            # 添加到已分配列表
            current_bubble['y'] = y_coord
            assigned_bubbles.append(current_bubble)
    
    # 合并single和overlap气泡
    df_combined = pd.concat([df_single, df_overlap], ignore_index=True)
    
    return df_combined

def find_optimal_y_with_gaussian(current_bubble, assigned_bubbles, mu, std, max_attempts=50):
    """
    根据高斯分布找到最优的y坐标(深度)，避免与已分配气泡重叠
    
    在三维坐标系中:
    - x: 表示图像宽度
    - y: 表示图像深度（本函数分配的坐标）
    - z: 表示图像高度
    
    参数:
        current_bubble: 当前气泡信息，包含x, z, radius, volume等
        assigned_bubbles: 已分配气泡列表
        mu: 高斯分布均值 (基于x坐标分布拟合)
        std: 高斯分布标准差 (基于x坐标分布拟合)
        max_attempts: 最大尝试次数
    
    返回:
        最优的y坐标（深度）
    """
    # 定义初始距离阈值 - 基于气泡体积
    volume = current_bubble['volume(mm^3)']
    radius = current_bubble['radius']
    
    # 体积越大，要求的距离阈值越大
    initial_threshold = radius * 2.5
    
    # 最小可接受阈值
    min_threshold = radius * 1.2
    
    # 当前阈值
    threshold = initial_threshold
    
    best_y = None
    min_overlap = float('inf')
    
    # 尝试多次采样
    for attempt in range(max_attempts):
        # 从高斯分布采样 - 使用与x相同的分布特性
        y_sample = np.random.normal(mu, std)
        y_sample = max(0, min(y_sample, DEPTH_RANGE))
        
        # 检查与已有气泡的重叠
        current_overlap = 0
        all_distances_ok = True
        
        for bubble in assigned_bubbles:
            # 计算3D欧氏距离
            distance = np.sqrt(
                (current_bubble['x'] - bubble['x'])**2 + 
                (y_sample - bubble['y'])**2 + 
                (current_bubble['z'] - bubble['z'])**2
            )
            
            # 两个气泡的半径和
            sum_of_radii = radius + bubble['radius']
            
            # 计算重叠程度 - 如果距离小于半径和，则有重叠
            if distance < sum_of_radii + threshold:
                current_overlap += sum_of_radii - distance
                all_distances_ok = False
        
        # 如果所有距离都满足阈值，直接返回
        if all_distances_ok:
            return y_sample
        
        # 更新最佳结果
        if current_overlap < min_overlap:
            min_overlap = current_overlap
            best_y = y_sample
    
    # 如果找不到满足条件的，降低阈值再试一次
    if threshold > min_threshold:
        threshold = max(min_threshold, threshold * 0.8)
        # 额外尝试
        for attempt in range(max_attempts):
            # 从高斯分布采样
            y_sample = np.random.normal(mu, std)
            y_sample = max(0, min(y_sample, DEPTH_RANGE))
            
            # 检查与已有气泡的重叠
            current_overlap = 0
            all_distances_ok = True
            
            for bubble in assigned_bubbles:
                # 计算3D欧氏距离
                distance = np.sqrt(
                    (current_bubble['x'] - bubble['x'])**2 + 
                    (y_sample - bubble['y'])**2 + 
                    (current_bubble['z'] - bubble['z'])**2
                )
                
                # 使用降低的阈值检查
                if distance < (radius + bubble['radius']) + threshold:
                    current_overlap += (radius + bubble['radius']) - distance
                    all_distances_ok = False
            
            # 如果所有距离都满足阈值，直接返回
            if all_distances_ok:
                return y_sample
            
            # 更新最佳结果
            if current_overlap < min_overlap:
                min_overlap = current_overlap
                best_y = y_sample
    
    # 如果仍然找不到理想的位置，返回最佳结果或随机生成
    return best_y if best_y is not None else np.random.normal(mu, std)

def visualize_bubbles_3d(df_3d, output_path):
    """创建3D可视化并保存"""
    # 创建PyVista对象
    plotter = pl.Plotter(window_size=[1920, 1440], off_screen=True)
    
    # 创建一个表示区域的边界框
    # 确保坐标系的一致性：x=宽度，y=深度，z=高度
    box = pl.Box(bounds=(0, IMAGE_WIDTH, 0, DEPTH_RANGE, 0, IMAGE_HEIGHT))
    plotter.add_mesh(box, style='wireframe', color='black', opacity=0.3)
    
    # 设置颜色映射
    try:
        # 适用于新版matplotlib
        cmap = plt.colormaps['viridis']
    except (AttributeError, ImportError):
        # 旧版matplotlib的回退方案
        cmap = cm.get_cmap('viridis')
    
    # 归一化气泡体积以用于着色
    volumes = df_3d['volume(mm^3)'].values
    vol_min, vol_max = volumes.min(), volumes.max()
    norm_volumes = (volumes - vol_min) / (vol_max - vol_min) if vol_max > vol_min else volumes * 0 + 0.5
    
    # 创建椭球体三维表面点的函数
    def create_ellipsoid_mesh(center, a, b, c, angle_rad, resolution=20):
        """创建椭球体网格点 - 简化版本
        
        Args:
            center: 椭球体中心(x, y, z)
            a: x轴半径
            b: z轴半径
            c: y轴半径（深度）
            angle_rad: x-z平面上的旋转角度（弧度）
            resolution: 网格分辨率
        
        Returns:
            椭球体网格对象
        """
        # 创建参数网格
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        
        # 生成标准椭球体坐标
        x = a * np.outer(np.cos(u), np.sin(v))
        y = c * np.outer(np.ones_like(u), np.cos(v))
        z = b * np.outer(np.sin(u), np.sin(v))
        
        # 应用旋转变换 (x-z平面上的旋转)
        x_rot = x * np.cos(angle_rad) - z * np.sin(angle_rad)
        z_rot = x * np.sin(angle_rad) + z * np.cos(angle_rad)
        
        # 移动到中心位置
        x_final = x_rot + center[0]
        y_final = y + center[1]
        z_final = z_rot + center[2]
        
        # 创建网格点坐标
        points = np.column_stack((x_final.flatten(), y_final.flatten(), z_final.flatten()))
        
        # 创建三角形连接点
        n_u, n_v = resolution, resolution
        
        # 创建网格索引
        faces = []
        for i in range(n_u-1):
            for j in range(n_v-1):
                p1 = i * n_v + j
                p2 = (i+1) * n_v + j
                p3 = (i+1) * n_v + (j+1)
                p4 = i * n_v + (j+1)
                
                # 添加两个三角形面
                faces.extend([3, p1, p2, p3])
                faces.extend([3, p1, p3, p4])
        
        # 创建PolyData对象
        mesh = pl.PolyData(points, faces=np.array(faces))
        return mesh
    
    # 创建一个列表存储所有气泡的网格
    bubble_meshes = []
    
    # 添加每个气泡
    for i, row in df_3d.iterrows():
        # 获取椭球体参数
        a, b, c = row['a'], row['b'], row['c']
        angle_rad = row['angle_rad']
        center = (row['x'], row['y'], row['z'])
        
        # 创建椭球体网格
        ellipsoid = create_ellipsoid_mesh(center, a, b, c, angle_rad)
        bubble_meshes.append(ellipsoid)  # 保存网格用于后续STL导出
        
        # 根据气泡类型和体积设置颜色
        color_idx = norm_volumes[i]
        color = cmap(color_idx)[:3]  # 获取RGB，排除Alpha
        
        # 根据气泡类型设置不同的透明度
        opacity = 0.8 if row['type'] == 'single' else 0.5
        
        # 添加椭球体到场景
        plotter.add_mesh(ellipsoid, color=color, opacity=opacity)
        
        # 添加气泡ID标签
        plotter.add_point_labels(
            points=[(row['x'], row['y'], row['z'])],
            labels=[str(int(row['bubble_id']))],
            point_size=1,
            font_size=8,
            shape_opacity=0.5
        )
    
    # 设置相机位置
    plotter.enable_parallel_projection()
    # 使用camera_position属性设置相机位置
    plotter.camera_position = [
        (IMAGE_WIDTH * 1, DEPTH_RANGE * 1, IMAGE_HEIGHT * 1),  # 相机位置
        (IMAGE_WIDTH/2, DEPTH_RANGE/2, IMAGE_HEIGHT/2),  # 焦点
        (0, 0, 1)  # 上方向
    ]
    plotter.reset_camera()  # 重置相机以确保视角生效
    
    # 添加坐标轴
    plotter.add_axes(xlabel='X (Width)', ylabel='Y (Depth)', zlabel='Z (Height)')
    
    # 添加标题
    plotter.add_text("Bubble 3D Reconstruction", font_size=20)
    plotter.camera.zoom(1.5)  # 扩大视图使模型充满屏幕
    # 保存图像
    screenshot_path = f"{output_path}/3d_visualization.png"
    plotter.screenshot(screenshot_path, window_size=[3840, 2880], return_img=False)
    print(f"3D visualization saved to {screenshot_path}")
    
    # 保存三个正交视角的截图
    # 设置正交投影
    
    # 视角1：正视图 (XZ平面)
    plotter.camera_position = 'yz'  # 使用预设视角
    plotter.reset_camera()  # 重置相机以确保视角生效
    # 调整缩放以充满整个视野
    plotter.camera.zoom(1.75)  # 扩大视图使模型充满屏幕
    front_view_path = f"{output_path}/3d_front_view.png"
    plotter.screenshot(front_view_path, window_size=[3840, 2880], return_img=False)
    print(f"正视图已保存至 {front_view_path}")
    
    # 视角2：侧视图 (YZ平面)
    plotter.camera_position = 'xz'  # 使用预设视角
    plotter.reset_camera()  # 重置相机以确保视角生效
    # 调整缩放以充满整个视野
    plotter.camera.zoom(1.75)  # 扩大视图使模型充满屏幕
    side_view_path = f"{output_path}/3d_side_view.png"
    plotter.screenshot(side_view_path, window_size=[3840, 2880], return_img=False)
    print(f"侧视图已保存至 {side_view_path}")
    
    # 视角3：顶视图 (XY平面)
    plotter.camera_position = 'xy'  # 使用预设视角
    plotter.reset_camera()  # 重置相机以确保视角生效
    # 调整缩放以充满整个视野
    plotter.camera.zoom(1.5)  # 扩大视图使模型充满屏幕
    top_view_path = f"{output_path}/3d_top_view.png"
    plotter.screenshot(top_view_path, window_size=[3840, 2880], return_img=False)
    print(f"顶视图已保存至 {top_view_path}")
    
    # 恢复透视投影
    # plotter.disable_parallel_projection()
    # 保存为STL文件
    # 从输出路径中提取帧号
    frame_num = int(os.path.basename(output_path).split('_')[1])
    # 计算时间（每帧0.0001秒）
    time_in_seconds = frame_num * 0.0001
    
    # 设置带有时间戳的STL文件路径 - 使用统一的stl_frames目录
    stl_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "stl_frames")
    os.makedirs(stl_dir, exist_ok=True)
    stl_path = f"{stl_dir}/{time_in_seconds:.4f}.stl"
    
    try:
        # 合并所有气泡网格
        combined_mesh = bubble_meshes[0]
        for mesh in bubble_meshes[1:]:
            combined_mesh = combined_mesh.merge(mesh)
        
        # 保存合并后的网格为 STL 文件
        combined_mesh.save(stl_path)
        print(f"3D model saved as STL to {stl_path}")
    except Exception as e:
        print(f"保存 STL 文件时出错: {str(e)}")
    
    # 关闭plotter
    plotter.close()

def export_bubble_info(df_3d, output_path, frame_num):
    """导出所有气泡的详细信息到单个CSV文件，文件名为当前帧号"""
    # 创建导出目录
    export_dir = os.path.join(output_path, "3-3Dbubble_positioning")
    os.makedirs(export_dir, exist_ok=True)
    
    # 按类型分组以便于统计
    df_single = df_3d[df_3d['type'] == 'single']
    df_overlap = df_3d[df_3d['type'] == 'overlap']
    
    # 导出所有气泡信息到单个文件，文件名为当前帧号
    all_bubbles_path = os.path.join(export_dir, f"frame_{frame_num:04d}.csv")
    df_3d.to_csv(all_bubbles_path, index=False)
    
    print(f"第 {frame_num} 帧的所有气泡信息已导出到 {all_bubbles_path}")
    return export_dir

def visualize_density_estimation(df_3d, output_path):
    """创建气泡分布密度估计图"""
    try:
        import numpy as np
        from scipy.stats import gaussian_kde
        
        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Bubble Density Distribution Estimation", fontsize=16)
        
        # XY平面密度估计 (俯视图) - X为宽度，Y为深度
        ax = axs[0, 0]
        xy_data = np.vstack([df_3d['x'], df_3d['y']])
        try:
            xy_kde = gaussian_kde(xy_data)
            
            # 创建网格点
            x_grid = np.linspace(0, IMAGE_WIDTH, 100)
            y_grid = np.linspace(0, DEPTH_RANGE, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            
            # 计算核密度估计
            Z = np.reshape(xy_kde(positions), X.shape)
            
            # 绘制等高线图
            cf = ax.contourf(X, Y, Z, cmap='Reds', alpha=0.8)
            ax.contour(X, Y, Z, colors='black', alpha=0.4, linewidths=0.5)
            plt.colorbar(cf, ax=ax, label='Density')
        except Exception as e:
            print(f"XY平面密度估计失败: {str(e)}")
            # 如果KDE失败，则绘制简单散点图
            ax.scatter(df_3d['x'], df_3d['y'], s=10, alpha=0.5, c='red')
            
        ax.set_xlabel('X (Width)')
        ax.set_ylabel('Y (Depth)')
        ax.set_title('Top View Density (XY Plane)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(0, DEPTH_RANGE)
        
        
        # XZ平面密度估计 (前视图) - X为宽度，Z为高度
        ax = axs[0, 1]
        xz_data = np.vstack([df_3d['x'], df_3d['z']])
        try:
            xz_kde = gaussian_kde(xz_data)
            
            # 创建网格点
            x_grid = np.linspace(0, IMAGE_WIDTH, 100)
            z_grid = np.linspace(0, IMAGE_HEIGHT, 100)
            X, Z = np.meshgrid(x_grid, z_grid)
            positions = np.vstack([X.ravel(), Z.ravel()])
            
            # 计算核密度估计
            Y = np.reshape(xz_kde(positions), X.shape)
            
            # 绘制等高线图
            cf = ax.contourf(X, Z, Y, cmap='Blues', alpha=0.8)
            ax.contour(X, Z, Y, colors='black', alpha=0.4, linewidths=0.5)
            plt.colorbar(cf, ax=ax, label='Density')
        except Exception as e:
            print(f"XZ平面密度估计失败: {str(e)}")
            # 如果KDE失败，则绘制简单散点图
            ax.scatter(df_3d['x'], df_3d['z'], s=10, alpha=0.5, c='blue')
            
        ax.set_xlabel('X (Width)')
        ax.set_ylabel('Z (Height)')
        ax.set_title('Front View Density (XZ Plane)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(0, IMAGE_HEIGHT)
        
        # YZ平面密度估计 (侧视图) - Y为深度，Z为高度
        ax = axs[1, 0]
        yz_data = np.vstack([df_3d['y'], df_3d['z']])
        try:
            yz_kde = gaussian_kde(yz_data)
            
            # 创建网格点
            y_grid = np.linspace(0, DEPTH_RANGE, 100)
            z_grid = np.linspace(0, IMAGE_HEIGHT, 100)
            Y, Z = np.meshgrid(y_grid, z_grid)
            positions = np.vstack([Y.ravel(), Z.ravel()])
            
            # 计算核密度估计
            X = np.reshape(yz_kde(positions), Y.shape)
            
            # 绘制等高线图
            cf = ax.contourf(Y, Z, X, cmap='Greens', alpha=0.8)
            ax.contour(Y, Z, X, colors='black', alpha=0.4, linewidths=0.5)
            plt.colorbar(cf, ax=ax, label='Density')
        except Exception as e:
            print(f"YZ平面密度估计失败: {str(e)}")
            # 如果KDE失败，则绘制简单散点图
            ax.scatter(df_3d['y'], df_3d['z'], s=10, alpha=0.5, c='green')
            
        ax.set_xlabel('Y (Depth)')
        ax.set_ylabel('Z (Height)')
        ax.set_title('Side View Density (YZ Plane)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, DEPTH_RANGE)
        ax.set_ylim(0, IMAGE_HEIGHT)
        
        # 体积分布直方图
        ax = axs[1, 1]
        volumes = df_3d['volume(mm^3)'].values
        
        # 计算合适的分箱数 - 增加分箱数使柱状图更窄
        bins = min(30, int(len(volumes) / 3) + 1)
        
        # 按气泡类型分组绘制直方图
        for bubble_type, color in zip(['single', 'overlap'], ['blue', 'red']):
            df_subset = df_3d[df_3d['type'] == bubble_type]
            if not df_subset.empty:
                ax.hist(df_subset['volume(mm^3)'], bins=bins, alpha=0.6, 
                       label=f"{bubble_type.capitalize()}", color=color, 
                       rwidth=0.7)  # 减少柱宽，使其更清晰
        
        ax.set_xlabel('Volume(mm³)')
        ax.set_ylabel('Frequency')
        ax.set_title('Bubble Volume Distribution')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        # 保存图表
        plt.tight_layout()
        density_path = f"{output_path}/density_estimation.png"
        plt.savefig(density_path, dpi=200)
        plt.close()
        print(f"Density estimation visualization saved to {density_path}")
        return True
    except Exception as e:
        print(f"密度估计可视化失败: {str(e)}")
        return False

def visualize_with_plotly_simple(df_3d, output_path):
    """
    使用Plotly创建简化版交互式3D可视化
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_file = f"{output_path}/Bubble Flow Visualization.html"
    
    # 分析气泡体积范围以进行颜色映射
    volumes = df_3d['volume(mm^3)'].values
    volume_min, volume_max = volumes.min(), volumes.max()
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        # 创建一个带有两个子图的图形：3D散点图和3D网格图
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{'type': 'scene'}], [{'type': 'scene'}]],
            subplot_titles=('Bubble Scatter Plot', 'Bubble Ellipsoid Mesh'),
            vertical_spacing=0.02  # 进一步减小垂直间距，几乎无间隙
        )
        
        def map_volume_to_size(volume):
            # 气泡体积转换为合理的点大小
            return 5 + (volume - volume_min) / (volume_max - volume_min) * 20
        
        def create_ellipsoid_mesh(center, a, b, c, angle_rad, resolution=20):
            """
            创建椭球体网格用于Plotly
            返回顶点和面
            """
            # 创建参数网格
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            u_grid, v_grid = np.meshgrid(u, v)
            
            # 生成椭球体的坐标
            x = a * np.cos(u_grid) * np.sin(v_grid)
            y = c * np.sin(u_grid) * np.sin(v_grid)
            z = b * np.cos(v_grid)
            
            # 旋转椭球体（绕y轴旋转，在x-z平面内旋转）
            x_rot = x * np.cos(angle_rad) - z * np.sin(angle_rad)
            z_rot = x * np.sin(angle_rad) + z * np.cos(angle_rad)
            
            # 平移到中心位置
            x_final = x_rot + center[0]
            y_final = y + center[1]  # y轴方向未旋转
            z_final = z_rot + center[2]
            
            # 创建顶点和面
            vertices = np.vstack([x_final.flatten(), y_final.flatten(), z_final.flatten()]).T
            
            # 创建三角形面
            faces = []
            for i in range(resolution-1):
                for j in range(resolution-1):
                    p1 = i * resolution + j
                    p2 = i * resolution + (j+1)
                    p3 = (i+1) * resolution + j
                    p4 = (i+1) * resolution + (j+1)
                    
                    # 两个三角形组成一个四边形
                    faces.append([p1, p2, p4])
                    faces.append([p1, p4, p3])
            
            return vertices, faces
        
        def map_volume_to_color(volume, min_vol=volume_min, max_vol=volume_max):
            """将气泡体积映射到颜色"""
            norm_value = (volume - min_vol) / (max_vol - min_vol)
            # 从蓝色到红色的渐变
            r = norm_value
            g = 0.2
            b = 1 - norm_value
            return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
        
        # 气泡类型到颜色和标识的映射
        bubble_types = {
            'single': {'symbol': 'circle', 'name': 'Single Bubble', 'marker_line_width': 1},
            'overlap': {'symbol': 'diamond', 'name': 'Overlapping Bubble', 'marker_line_width': 2}
        }
        
        # 按气泡类型分组添加散点
        for bubble_type, type_data in df_3d.groupby('type'):
            type_info = bubble_types.get(bubble_type, {'symbol': 'circle', 'name': bubble_type, 'marker_line_width': 1})
            
            # 为散点图添加不同颜色的气泡
            fig.add_trace(
                go.Scatter3d(
                    x=type_data['x'],
                    y=type_data['y'],
                    z=type_data['z'],
                    mode='markers',
                    marker=dict(
                        size=[map_volume_to_size(v) for v in type_data['volume(mm^3)']],
                        color=[map_volume_to_color(v) for v in type_data['volume(mm^3)']],
                        symbol=type_info['symbol'],
                        opacity=0.8,
                        line=dict(width=type_info['marker_line_width'], color='white')
                    ),
                    text=[f"ID: {i}<br>Volume: {v:.2f} mm³<br>Coordinates: ({x:.1f}, {y:.1f}, {z:.1f})" 
                          for i, v, x, y, z in zip(
                              type_data['bubble_id'], 
                              type_data['volume(mm^3)'], 
                              type_data['x'], 
                              type_data['y'], 
                              type_data['z']
                          )],
                    hoverinfo='text',
                    hovertemplate='%{text}<extra></extra>',
                    name=type_info['name']
                ),
                row=1, col=1
            )
            
            # 为网格图添加椭球体
            for i, row in type_data.iterrows():
                vertices, faces = create_ellipsoid_mesh(
                    center=(row['x'], row['y'], row['z']),
                    a=row['a'],
                    b=row['b'],
                    c=row['c'],
                    angle_rad=row['angle_rad'],
                    resolution=20  # 降低分辨率以提高性能
                )
                
                # 添加椭球体的网格
                color = map_volume_to_color(row['volume(mm^3)'])
                opacity = 0.7 if row['type'] == 'single' else 0.5
                
                fig.add_trace(
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=[face[0] for face in faces],
                        j=[face[1] for face in faces],
                        k=[face[2] for face in faces],
                        color=color,
                        opacity=opacity,
                        hoverinfo='text',
                        hovertext=f"ID: {row['bubble_id']}<br>Volume: {row['volume(mm^3)']:.2f} mm³<br>Type: {row['type']}",
                        showscale=False,
                        lighting=dict(
                            ambient=0.3,
                            diffuse=0.8,
                            roughness=0.5,
                            specular=0.8,
                            fresnel=0.1
                        ),
                        name=f"Bubble {row['bubble_id']}"
                    ),
                    row=2, col=1
                )
        
        x_range = [0, IMAGE_WIDTH]
        y_range = [0, DEPTH_RANGE]
        z_range = [0, IMAGE_HEIGHT]
        
        # 计算实际的物理比例
        x_scale = IMAGE_WIDTH
        y_scale = DEPTH_RANGE
        z_scale = IMAGE_HEIGHT
        
        # 设置两个3D场景的属性
        for i in range(1, 3):
            fig.update_scenes(
                aspectmode='manual',  # 手动设置比例以确保与实际物理尺寸一致
                xaxis=dict(
                    title=dict(text='X Axis (Width)', font=dict(size=12, family="Arial, sans-serif")),
                    range=x_range,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgb(245, 245, 245)'  # 更浅的背景色
                ),
                yaxis=dict(
                    title=dict(text='Y Axis (Depth)', font=dict(size=12, family="Arial, sans-serif")),
                    range=y_range,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgb(245, 245, 245)'  # 更浅的背景色
                ),
                zaxis=dict(
                    title=dict(text='Z Axis (Height)', font=dict(size=12, family="Arial, sans-serif")),
                    range=z_range,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgb(245, 245, 245)'  # 更浅的背景色
                ),
                aspectratio=dict(x=x_scale, y=y_scale, z=z_scale),  # 使用实际物理比例
                camera=dict(
                    up=dict(x=0, y=0, z=1),  # z轴向上
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=-1.5, z=1.5),   # 初始视角
                    projection=dict(type='orthographic')
                ),
                row=i, col=1
            )
        
        # 更新布局 - 调整为两行一列的布局尺寸
        fig.update_layout(
            scene=dict(
                aspectratio=dict(x=1280, y=1280, z=800),  # 设置坐标轴的比例
                aspectmode='manual',  # 手动设置比例模式
                xaxis=dict(
                    title=dict(text='X Axis (Width)', font=dict(size=14, family="Arial, sans-serif")),
                    range=x_range,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgb(245, 245, 245)'  # 更浅的背景色
                ),
                yaxis=dict(
                    title=dict(text='Y Axis (Depth)', font=dict(size=14, family="Arial, sans-serif")),
                    range=y_range,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgb(245, 245, 245)'  # 更浅的背景色
                ),
                zaxis=dict(
                    title=dict(text='Z Axis (Height)', font=dict(size=14, family="Arial, sans-serif")),
                    range=z_range,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgb(245, 245, 245)'  # 更浅的背景色
                ),
                camera=dict(
                    up=dict(x=0, y=0, z=1),  # z轴向上
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=-1.5, z=1.5)  # 初始视角
                )
            ),
            title=dict(
                text='Bubble Flow 3D Visualization',
                font=dict(size=24, family="Arial, sans-serif", color='rgb(30, 30, 30)'),
                y=0.98  # 将标题往上移，给图表留出更多空间
            ),
            legend=dict(
                itemsizing='constant',
                font=dict(size=14, family="Arial, sans-serif"),
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=30, r=30, t=80, b=120),  # 增加底部边距，为颜色条留出空间
            template='plotly_white',
            autosize=False,
            width=1400,
            height=2500,  # 保持高度，确保可视区域足够大
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',  # 移除背景网格
            paper_bgcolor='rgba(0,0,0,0)'  # 透明背景
        )
        
        # 创建和添加用于颜色条的虚拟散点图
        # 创建更详细的刻度值
        tick_count = 7  # 增加刻度数量
        tick_vals = np.linspace(0, 1, tick_count)
        tick_text = [f"{val:.2f}" for val in np.linspace(volume_min, volume_max, tick_count)]
        
        # 添加最小值和最大值标签
        min_vol_text = f"Minimum: {volume_min:.2f} mm³"
        max_vol_text = f"Maximum: {volume_max:.2f} mm³"
        
        colorbar_trace = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                colorscale=[[0, 'rgb(0, 50, 255)'], [1, 'rgb(255, 50, 0)']],  # 从蓝色到红色，与我们的映射函数一致
                showscale=True,
                colorbar=dict(
                    title="Bubble Volume (mm³)",
                    titleside="right",
                    titlefont=dict(size=14),
                    thickness=20,
                    len=0.8,  # 颜色条长度
                    yanchor="bottom",
                    y=0.01,  # 将位置向下移动
                    xanchor="center",
                    x=0.5,   # 水平居中
                    orientation="h",  # 水平方向
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    tickmode="array",
                    tickfont=dict(size=12),
                    title_text=f"{min_vol_text} | {max_vol_text}",
                    title_font=dict(size=12)
                ),
                cmin=volume_min,
                cmax=volume_max
            ),
            hoverinfo='none'
        )
        fig.add_trace(colorbar_trace)
        
        # 保存HTML文件
        fig.write_html(output_file, include_plotlyjs='cdn', config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'Bubble Flow Visualization',
                'height': 2800,
                'width': 1400,
                'scale': 2
            }
        })
        
        print(f"单一3D可视化已保存到 {output_file}")
    except Exception as e:
        print(f"创建Plotly可视化失败: {str(e)}")
        raise

def main():
    # 定义输入和输出路径
    input_dir = r"C:\codebase\BFRM\results\yolo11l-obb\1-bubble_csv"  # 使用yolo11l-obb
    output_dir = r"C:\codebase\BFRM\results\yolo11l-obb"
    
    input_dir = r"C:\codebase\BFRM\results\RB_bubble_flow1\1-bubble_csv"
    output_dir = r"C:\codebase\BFRM\results\RB_bubble_flow1"
    
    input_dir = r"C:\codebase\BFRM\results\RB_bubble_flow2\1-bubble_csv"
    output_dir = r"C:\codebase\BFRM\results\RB_bubble_flow2"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 注释掉处理所有帧的代码，仅处理第一帧
    # flow_3d_dir = process_all_frames(input_dir, output_dir)
    
    # 读取第一帧数据用于处理和可视化
    first_frame_path = os.path.join(input_dir, "frame_0001.csv")
    frame_num = 1  # 当前处理的帧号
    
    if not os.path.exists(first_frame_path):
        print(f"错误: 文件 {first_frame_path} 不存在")
        return
    
    print(f"开始处理第一帧: {first_frame_path}")
    
    # 读取和处理CSV
    df = read_bubble_csv(first_frame_path)
    
    # 转换坐标
    df_3d = convert_coordinates(df)
    
    # 分配y坐标（深度）
    df_3d = assign_y_coordinates(df_3d)
      
    # 导出气泡详细信息到单个CSV文件，并传递帧号
    bubble_info_dir = export_bubble_info(df_3d, output_dir, frame_num)
    print(f"第 {frame_num} 帧的气泡详细信息已导出到: {bubble_info_dir}")
    
    # 创建额外的可视化目录
    visualizations_dir = os.path.join(output_dir, f"3-simple_visualization/frame_{frame_num:04d}")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # 可视化选项 - 添加Plotly多视角可视化
    visualization_methods = {
        "PyVista 3D": lambda: visualize_bubbles_3d(df_3d, visualizations_dir),
        "Density Estimation": lambda: visualize_density_estimation(df_3d, visualizations_dir),
        "Plotly Single": lambda: visualize_with_plotly_simple(df_3d, visualizations_dir)
    }
    
    # 执行所有可视化方法
    for name, method in visualization_methods.items():
        print(f"执行{name}可视化...")
        try:
            method()
            print(f"{name}可视化完成")
        except Exception as e:
            print(f"{name}可视化失败: {str(e)}")
    
    print(f"第一帧处理完成，所有可视化文件已保存到: {visualizations_dir}")
    return visualizations_dir

if __name__ == "__main__":
    main()

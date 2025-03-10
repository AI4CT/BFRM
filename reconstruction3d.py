import os
import numpy as np
import pandas as pd
import pyvista as pl
from tqdm import tqdm
import matplotlib.cm as cm
from scipy import stats
import random

# 定义常量
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 800
DEPTH_RANGE = 1280

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
    """转换坐标系，从图像坐标到3D空间坐标"""
    # 复制数据框以避免修改原始数据
    df_3d = df.copy()
    
    # 将图像坐标转换为3D坐标
    # x 保持不变
    # y 将在后续步骤中分配
    # 图像的y坐标转换为z坐标，并翻转方向（图像原点在左上角）
    df_3d['z'] = IMAGE_HEIGHT - df_3d['y']
    
    return df_3d

def assign_y_coordinates(df_3d):
    """为气泡分配y坐标（深度）"""
    # 按类型和置信度排序
    df_single = df_3d[df_3d['type'] == 'single'].copy()
    df_overlap = df_3d[df_3d['type'] == 'overlap'].copy()
    
    # 计算x坐标的统计特性
    x_values = df_3d['x'].values
    # 拟合x坐标的正态分布
    mu_x, std_x = stats.norm.fit(x_values)
    print(f"X坐标分布拟合结果: 均值={mu_x:.2f}, 标准差={std_x:.2f}")
    
    # 确保标准差不会太小，以保证充分的空间分布
    std_x = max(std_x, IMAGE_WIDTH / 10)
    
    # 定义气泡半径计算函数
    def calc_bubble_radius(row):
        # 使用宽度和高度的平均值的一半作为气泡半径
        return (row['width'] + row['height']) / 4
    
    # 为single气泡分配y坐标 - 使用与x相同的高斯分布
    if not df_single.empty:
        # 确保y坐标在合理范围内
        y_coords = []
        for _ in range(len(df_single)):
            # 从与x相同的正态分布中采样y坐标
            y = np.random.normal(mu_x, std_x)
            # 确保y在[0, DEPTH_RANGE]范围内
            y = max(0, min(y, DEPTH_RANGE))
            y_coords.append(y)
        
        df_single['y'] = y_coords
    
    # 添加气泡半径列，用于后续计算
    df_single['radius'] = df_single.apply(calc_bubble_radius, axis=1)
    
    # 为overlap气泡分配y坐标
    if not df_overlap.empty:
        # 添加气泡半径列
        df_overlap['radius'] = df_overlap.apply(calc_bubble_radius, axis=1)
        
        # 已分配气泡的3D位置
        assigned_bubbles = []
        if not df_single.empty:
            for _, row in df_single.iterrows():
                assigned_bubbles.append({
                    'x': row['x'],
                    'y': row['y'],
                    'z': row['z'],
                    'radius': row['radius'],
                    'volume': row['volume(mm^3)']
                })
        
        # 逐个处理overlap气泡
        for idx, row in df_overlap.iterrows():
            current_bubble = {
                'x': row['x'],
                'z': row['z'],
                'radius': row['radius'],
                'volume': row['volume(mm^3)']
            }
            
            # 尝试为当前气泡找到合适的y坐标
            y_coord = find_optimal_y_with_gaussian(
                current_bubble, 
                assigned_bubbles, 
                mu_x, 
                std_x
            )
            
            df_overlap.at[idx, 'y'] = y_coord
            
            # 将当前气泡添加到已分配列表
            current_bubble['y'] = y_coord
            assigned_bubbles.append(current_bubble)
    
    # 合并single和overlap气泡
    df_combined = pd.concat([df_single, df_overlap], ignore_index=True)
    
    # 删除临时列
    if 'radius' in df_combined.columns:
        df_combined.drop('radius', axis=1, inplace=True)
    
    return df_combined

def find_optimal_y_with_gaussian(current_bubble, assigned_bubbles, mu, std, max_attempts=50):
    """
    根据高斯分布找到最优的y坐标，避免与已分配气泡重叠
    
    参数:
        current_bubble: 当前气泡信息，包含x, z, radius, volume
        assigned_bubbles: 已分配气泡列表
        mu: 高斯分布均值
        std: 高斯分布标准差
        max_attempts: 最大尝试次数
    
    返回:
        最优的y坐标
    """
    import numpy as np
    
    # 如果没有已分配气泡，直接从高斯分布采样
    if not assigned_bubbles:
        y = np.random.normal(mu, std)
        return max(0, min(y, DEPTH_RANGE))
    
    # 定义初始距离阈值 - 基于气泡体积
    volume = current_bubble['volume']
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
        # 从高斯分布采样
        y_sample = np.random.normal(mu, std)
        y_sample = max(0, min(y_sample, DEPTH_RANGE))
        
        # 计算与所有已分配气泡的重叠程度
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
        for attempt in range(max_attempts // 2):
            y_sample = np.random.normal(mu, std)
            y_sample = max(0, min(y_sample, DEPTH_RANGE))
            
            current_overlap = 0
            all_distances_ok = True
            
            for bubble in assigned_bubbles:
                distance = np.sqrt(
                    (current_bubble['x'] - bubble['x'])**2 + 
                    (y_sample - bubble['y'])**2 + 
                    (current_bubble['z'] - bubble['z'])**2
                )
                
                sum_of_radii = radius + bubble['radius']
                
                if distance < sum_of_radii + threshold:
                    current_overlap += sum_of_radii - distance
                    all_distances_ok = False
            
            if all_distances_ok:
                return y_sample
            
            if current_overlap < min_overlap:
                min_overlap = current_overlap
                best_y = y_sample
    
    # 如果仍然找不到理想的位置，返回最佳结果
    return best_y if best_y is not None else np.random.normal(mu, std)

def visualize_bubbles_3d(df_3d, output_path):
    """创建3D可视化并保存"""
    # 创建PyVista对象
    plotter = pl.Plotter(window_size=[1024, 768], off_screen=True)
    
    # 创建一个表示区域的边界框
    box = pl.Box(bounds=(0, IMAGE_WIDTH, 0, DEPTH_RANGE, 0, IMAGE_HEIGHT))
    plotter.add_mesh(box, style='wireframe', color='black', opacity=0.3)
    
    # 设置颜色映射
    try:
        # 适用于新版matplotlib
        import matplotlib.pyplot as plt
        cmap = plt.colormaps['viridis']
    except (AttributeError, ImportError):
        # 旧版matplotlib的回退方案
        cmap = cm.get_cmap('viridis')
    
    # 归一化气泡体积以用于着色
    volumes = df_3d['volume(mm^3)'].values
    vol_min, vol_max = volumes.min(), volumes.max()
    norm_volumes = (volumes - vol_min) / (vol_max - vol_min) if vol_max > vol_min else volumes * 0 + 0.5
    
    # 添加每个气泡
    for i, row in df_3d.iterrows():
        # 计算半径（基于宽度和高度的平均值的一半）
        radius = (row['width'] + row['height']) / 4
        
        # 创建球体表示气泡
        sphere = pl.Sphere(radius=radius, center=(row['x'], row['y'], row['z']))
        
        # 根据气泡类型和体积设置颜色
        color_idx = norm_volumes[i]
        color = cmap(color_idx)[:3]  # 获取RGB，排除Alpha
        
        # 根据气泡类型设置不同的透明度
        opacity = 0.8 if row['type'] == 'single' else 0.5
        
        # 添加球体到场景
        plotter.add_mesh(sphere, color=color, opacity=opacity)
        
        # 添加气泡ID标签
        plotter.add_point_labels(
            points=[(row['x'], row['y'], row['z'])],
            labels=[str(int(row['bubble_id']))],
            point_size=1,
            font_size=8,
            shape_opacity=0.5
        )
    
    # 设置相机位置
    plotter.camera.position = (IMAGE_WIDTH * 1.5, DEPTH_RANGE * 1.5, IMAGE_HEIGHT * 1.5)
    plotter.camera.focal_point = (IMAGE_WIDTH/2, DEPTH_RANGE/2, IMAGE_HEIGHT/2)
    plotter.camera.up = (0, 0, 1)
    
    # 添加坐标轴
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    # 添加标题
    plotter.add_text("Bubble 3D Reconstruction", font_size=20)
    
    # 保存图像
    screenshot_path = f"{output_path}/3d_visualization.png"
    plotter.screenshot(screenshot_path)
    print(f"3D visualization saved to {screenshot_path}")
    
    # 移除HTML导出功能，避免依赖问题
    # plotter.export_html(f"{output_path}/3d_visualization.html")
    
    # 关闭plotter
    plotter.close()

def save_3d_data(df_3d, output_path):
    """保存处理后的3D数据"""
    try:
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 使用os.path.join构建完整的文件路径
        output_file = os.path.join(output_path, "bubbles_3d.csv")
        
        # 保存数据
        df_3d.to_csv(output_file, index=False, encoding='utf-8')
        print(f"3D数据已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存3D数据时出错: {str(e)}")
        return False

def visualize_with_matplotlib(df_3d, output_path):
    """使用Matplotlib创建3D可视化（作为备选方案）"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 创建图形对象
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置坐标轴范围
    ax.set_xlim(0, IMAGE_WIDTH)
    ax.set_ylim(0, DEPTH_RANGE)
    ax.set_zlim(0, IMAGE_HEIGHT)
    
    # 按照类型分组
    df_single = df_3d[df_3d['type'] == 'single']
    df_overlap = df_3d[df_3d['type'] == 'overlap']
    
    # 将体积映射到气泡大小
    def map_volume_to_size(volume):
        return 20 + volume / 10  # 简单的线性映射，可以根据需要调整
    
    # 绘制single气泡
    scatter1 = ax.scatter(
        df_single['x'], 
        df_single['y'], 
        df_single['z'],
        s=df_single['volume(mm^3)'].apply(map_volume_to_size),
        c='blue',
        alpha=0.7,
        marker='o',
        label='Single Bubbles'
    )
    
    # 绘制overlap气泡
    scatter2 = ax.scatter(
        df_overlap['x'], 
        df_overlap['y'], 
        df_overlap['z'],
        s=df_overlap['volume(mm^3)'].apply(map_volume_to_size),
        c='red',
        alpha=0.5,
        marker='o',
        label='Overlap Bubbles'
    )
    
    # 为部分气泡添加ID标签
    for i, row in df_3d.iterrows():
        if i % 10 == 0:  # 每10个气泡添加一个标签，避免太拥挤
            ax.text(row['x'], row['y'], row['z'], f"{int(row['bubble_id'])}", 
                   fontsize=8, color='black')
    
    # 添加图例
    ax.legend()
    
    # 添加标题
    plt.title("Bubble 3D Reconstruction (Matplotlib)")
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 保存图像
    plt.tight_layout()
    mpl_path = f"{output_path}/3d_visualization_mpl.png"
    plt.savefig(mpl_path, dpi=200)
    plt.close()
    print(f"Matplotlib 3D visualization saved to {mpl_path}")

def visualize_with_plotly(df_3d, output_path):
    """使用Plotly创建交互式3D可视化"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import numpy as np
        
        # 按类型分组
        df_single = df_3d[df_3d['type'] == 'single']
        df_overlap = df_3d[df_3d['type'] == 'overlap']
        
        # 创建3D散点图
        fig = go.Figure()
        
        # 为single气泡添加散点
        fig.add_trace(go.Scatter3d(
            x=df_single['x'],
            y=df_single['y'],
            z=df_single['z'],
            mode='markers',
            name='Single Bubbles',
            marker=dict(
                size=df_single['volume(mm^3)'].apply(lambda v: np.cbrt(v) * 1.5),
                color='blue',
                opacity=0.7,
                symbol='circle',
                line=dict(color='black', width=0.5)
            ),
            text=df_single['bubble_id'].apply(lambda id: f"ID: {id}<br>Volume: {df_single.loc[df_single['bubble_id']==id, 'volume(mm^3)'].values[0]:.2f} mm³"),
            hoverinfo='text'
        ))
        
        # 为overlap气泡添加散点
        fig.add_trace(go.Scatter3d(
            x=df_overlap['x'],
            y=df_overlap['y'],
            z=df_overlap['z'],
            mode='markers',
            name='Overlap Bubbles',
            marker=dict(
                size=df_overlap['volume(mm^3)'].apply(lambda v: np.cbrt(v) * 1.5),
                color='red',
                opacity=0.6,
                symbol='circle',
                line=dict(color='black', width=0.5)
            ),
            text=df_overlap['bubble_id'].apply(lambda id: f"ID: {id}<br>Volume: {df_overlap.loc[df_overlap['bubble_id']==id, 'volume(mm^3)'].values[0]:.2f} mm³"),
            hoverinfo='text'
        ))
        
        # 设置布局
        fig.update_layout(
            title="Interactive 3D Bubble Reconstruction",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0, y=1),
            template='plotly_white'
        )
        
        # 保存为HTML文件
        html_path = f"{output_path}/3d_interactive_visualization.html"
        fig.write_html(html_path)
        print(f"Interactive 3D visualization saved to {html_path}")
        
        # 创建多视图展示
        fig_views = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=['Front View', 'Top View', 'Side View', 'Isometric View']
        )
        
        # 所有气泡数据
        for df, color, name in [(df_single, 'blue', 'Single'), (df_overlap, 'red', 'Overlap')]:
            # 正视图 (Front View)
            fig_views.add_trace(
                go.Scatter3d(
                    x=df['x'], y=df['y'], z=df['z'],
                    mode='markers',
                    marker=dict(size=df['volume(mm^3)'].apply(lambda v: np.cbrt(v) * 1.5),
                                color=color, opacity=0.7),
                    name=name
                ),
                row=1, col=1
            )
            
            # 俯视图 (Top View)
            fig_views.add_trace(
                go.Scatter3d(
                    x=df['x'], y=df['y'], z=df['z'],
                    mode='markers',
                    marker=dict(size=df['volume(mm^3)'].apply(lambda v: np.cbrt(v) * 1.5),
                                color=color, opacity=0.7),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 侧视图 (Side View)
            fig_views.add_trace(
                go.Scatter3d(
                    x=df['x'], y=df['y'], z=df['z'],
                    mode='markers',
                    marker=dict(size=df['volume(mm^3)'].apply(lambda v: np.cbrt(v) * 1.5),
                                color=color, opacity=0.7),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 等轴测图 (Isometric View)
            fig_views.add_trace(
                go.Scatter3d(
                    x=df['x'], y=df['y'], z=df['z'],
                    mode='markers',
                    marker=dict(size=df['volume(mm^3)'].apply(lambda v: np.cbrt(v) * 1.5),
                                color=color, opacity=0.7),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 设置各个视图的摄像机位置
        fig_views.update_layout(
            scene=dict(
                camera=dict(eye=dict(x=0, y=-2, z=0)),  # 正视图
                aspectmode='cube'
            ),
            scene2=dict(
                camera=dict(eye=dict(x=0, y=0, z=2)),  # 俯视图
                aspectmode='cube'
            ),
            scene3=dict(
                camera=dict(eye=dict(x=2, y=0, z=0)),  # 侧视图
                aspectmode='cube'
            ),
            scene4=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # 等轴测图
                aspectmode='cube'
            ),
            title="Multi-View 3D Bubble Visualization",
            height=1000,
            width=1000
        )
        
        # 保存多视图HTML文件
        multi_view_path = f"{output_path}/3d_multi_view_visualization.html"
        fig_views.write_html(multi_view_path)
        print(f"Multi-view 3D visualization saved to {multi_view_path}")
        
        # 创建气泡体积分布热图
        fig_heatmap = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
        
        # 创建体积连续色标
        all_volumes = df_3d['volume(mm^3)'].values
        volume_color = (all_volumes - all_volumes.min()) / (all_volumes.max() - all_volumes.min())
        
        fig_heatmap.add_trace(
            go.Scatter3d(
                x=df_3d['x'],
                y=df_3d['y'],
                z=df_3d['z'],
                mode='markers',
                marker=dict(
                    size=df_3d['volume(mm^3)'].apply(lambda v: np.cbrt(v) * 1.5),
                    color=volume_color,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Volume (mm³)"),
                    showscale=True
                ),
                text=df_3d.apply(lambda row: f"ID: {row['bubble_id']}<br>Type: {row['type']}<br>Volume: {row['volume(mm^3)']:.2f} mm³", axis=1),
                hoverinfo='text'
            )
        )
        
        fig_heatmap.update_layout(
            title="Bubble Volume Distribution in 3D Space",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # 保存体积分布热图
        heatmap_path = f"{output_path}/3d_volume_heatmap.html"
        fig_heatmap.write_html(heatmap_path)
        print(f"Volume heatmap visualization saved to {heatmap_path}")
        
        return True
    except ImportError as e:
        print(f"Plotly可视化失败: 缺少库 {str(e)}")
        return False
    except Exception as e:
        print(f"Plotly可视化失败: {str(e)}")
        return False

def visualize_2d_projections(df_3d, output_path):
    """创建2D投影平面视图"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("2D Projections of 3D Bubble Distribution", fontsize=16)
        
        # 定义不同类型气泡的颜色和透明度
        colors = {'single': 'blue', 'overlap': 'red'}
        alpha = {'single': 0.7, 'overlap': 0.5}
        
        # XY平面投影 (俯视图)
        ax = axs[0, 0]
        for bubble_type in ['single', 'overlap']:
            df_subset = df_3d[df_3d['type'] == bubble_type]
            ax.scatter(
                df_subset['x'], 
                df_subset['y'],
                s=df_subset['volume(mm^3)'] / 3,
                c=colors[bubble_type],
                alpha=alpha[bubble_type],
                edgecolors='black',
                linewidths=0.2,
                label=f"{bubble_type.capitalize()} Bubbles"
            )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Top View (XY Plane)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(0, DEPTH_RANGE)
        
        # XZ平面投影 (前视图)
        ax = axs[0, 1]
        for bubble_type in ['single', 'overlap']:
            df_subset = df_3d[df_3d['type'] == bubble_type]
            ax.scatter(
                df_subset['x'], 
                df_subset['z'],
                s=df_subset['volume(mm^3)'] / 3,
                c=colors[bubble_type],
                alpha=alpha[bubble_type],
                edgecolors='black',
                linewidths=0.2,
                label=f"{bubble_type.capitalize()} Bubbles"
            )
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('Front View (XZ Plane)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(0, IMAGE_HEIGHT)
        
        # YZ平面投影 (侧视图)
        ax = axs[1, 0]
        for bubble_type in ['single', 'overlap']:
            df_subset = df_3d[df_3d['type'] == bubble_type]
            ax.scatter(
                df_subset['y'], 
                df_subset['z'],
                s=df_subset['volume(mm^3)'] / 3,
                c=colors[bubble_type],
                alpha=alpha[bubble_type],
                edgecolors='black',
                linewidths=0.2,
                label=f"{bubble_type.capitalize()} Bubbles"
            )
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title('Side View (YZ Plane)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, DEPTH_RANGE)
        ax.set_ylim(0, IMAGE_HEIGHT)
        
        # 体积分布热图
        ax = axs[1, 1]
        scatter = ax.scatter(
            df_3d['x'], 
            df_3d['z'],
            s=df_3d['volume(mm^3)'] / 2,
            c=df_3d['volume(mm^3)'],
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidths=0.2
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('Volume Distribution (XZ Plane)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(0, IMAGE_HEIGHT)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Volume (mm³)')
        
        # 添加图例 (只在第一个子图添加)
        axs[0, 0].legend(loc='upper right')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图表
        projections_path = f"{output_path}/2d_projections.png"
        plt.savefig(projections_path, dpi=200)
        plt.close()
        print(f"2D projections saved to {projections_path}")
        
        return True
    except Exception as e:
        print(f"2D投影可视化失败: {str(e)}")
        return False
        
def visualize_density_estimation(df_3d, output_path):
    """创建气泡分布密度估计图"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import gaussian_kde
        
        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Bubble Density Distribution", fontsize=16)
        
        # XY平面密度估计 (俯视图)
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
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Top View Density (XY Plane)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(0, DEPTH_RANGE)
        
        # XZ平面密度估计 (前视图)
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
            
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('Front View Density (XZ Plane)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(0, IMAGE_HEIGHT)
        
        # YZ平面密度估计 (侧视图)
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
            
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title('Side View Density (YZ Plane)')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, DEPTH_RANGE)
        ax.set_ylim(0, IMAGE_HEIGHT)
        
        # 体积分布直方图
        ax = axs[1, 1]
        volumes = df_3d['volume(mm^3)']
        
        # 计算合适的分箱数
        bins = min(20, int(len(volumes) / 5) + 1)
        
        # 按气泡类型分组绘制直方图
        for bubble_type, color in [('single', 'blue'), ('overlap', 'red')]:
            subset = df_3d[df_3d['type'] == bubble_type]['volume(mm^3)']
            if len(subset) > 0:
                ax.hist(subset, bins=bins, alpha=0.6, color=color, 
                        label=f"{bubble_type.capitalize()} Bubbles", edgecolor='black')
        
        ax.set_xlabel('Volume (mm³)')
        ax.set_ylabel('Count')
        ax.set_title('Bubble Volume Distribution')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图表
        density_path = f"{output_path}/density_estimation.png"
        plt.savefig(density_path, dpi=200)
        plt.close()
        print(f"Density estimation saved to {density_path}")
        
        return True
    except Exception as e:
        print(f"密度估计可视化失败: {str(e)}")
        return False

def main():
    # 定义输入和输出路径
    input_dir = r"C:\codebase\BFRM\results\yolo11l-obb\bubble_csv"  # 使用yolo11l-obb
    output_dir = r"C:\codebase\BFRM\results\yolo11l-obb"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取第一帧数据
    first_frame_path = os.path.join(input_dir, "frame_0000.csv")
    
    if not os.path.exists(first_frame_path):
        print(f"错误: 文件 {first_frame_path} 不存在")
        return
    
    # 读取和处理CSV
    df = read_bubble_csv(first_frame_path)
    
    # 转换坐标
    df_3d = convert_coordinates(df)
    
    # 分配y坐标（深度）
    df_3d = assign_y_coordinates(df_3d)
    
    # 保存3D数据
    save_3d_data(df_3d, output_dir)
    
    # 创建额外的可视化目录
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # 可视化选项
    visualization_methods = {
        "PyVista 3D": lambda: visualize_bubbles_3d(df_3d, visualizations_dir),
        "Matplotlib 3D": lambda: visualize_with_matplotlib(df_3d, visualizations_dir),
        "Plotly Interactive 3D": lambda: visualize_with_plotly(df_3d, visualizations_dir),
        "2D Projections": lambda: visualize_2d_projections(df_3d, visualizations_dir),
        "Density Estimation": lambda: visualize_density_estimation(df_3d, visualizations_dir)
    }
    
    # 执行所有可视化方法
    for name, visualization_method in visualization_methods.items():
        print(f"\n生成 {name} 可视化...")
        try:
            visualization_method()
        except Exception as e:
            print(f"{name} 可视化失败: {str(e)}")
    
    print("\n三维重建和可视化完成。所有可视化文件保存在:", visualizations_dir)

if __name__ == "__main__":
    main()

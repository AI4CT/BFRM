import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

# 定义输入文件夹
input_folder = r"C:\codebase\BFRM\results\yolo11l-obb\3d_reconstructed"
output_folder = r"C:\codebase\BFRM\results\bubble_analysis"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 读取所有CSV文件
def load_all_frames():
    """加载所有帧的数据"""
    csv_files = sorted(glob.glob(os.path.join(input_folder, "frame_*.csv")))
    frames_data = []
    
    for file_path in csv_files:
        try:
            frame_num = int(os.path.basename(file_path).split('_')[1].split('.')[0])
            df = pd.read_csv(file_path, encoding='gbk')
            df['frame'] = frame_num
            frames_data.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
    
    return frames_data

# 分析气泡轨迹
def analyze_bubble_trajectories(frames_data):
    """分析气泡的运动轨迹"""
    # 合并所有帧的数据
    all_data = pd.concat(frames_data, ignore_index=True)
    
    # 按气泡ID和帧号排序
    all_data = all_data.sort_values(['bubble_id', 'frame'])
    
    # 统计每个气泡出现的帧数
    bubble_counts = all_data['bubble_id'].value_counts()
    
    # 获取出现次数最多的前N个气泡
    top_bubbles = bubble_counts.nlargest(10).index.tolist()
    
    # 提取这些气泡的轨迹
    trajectories = {}
    for bubble_id in top_bubbles:
        bubble_data = all_data[all_data['bubble_id'] == bubble_id]
        trajectories[bubble_id] = bubble_data
    
    return trajectories, all_data

# 绘制气泡轨迹的3D图
def plot_bubble_trajectories(trajectories):
    """绘制气泡运动轨迹的3D图"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    
    for i, (bubble_id, data) in enumerate(trajectories.items()):
        x = data['x'].values
        y = data['y'].values
        z = data['z'].values
        frames = data['frame'].values
        
        # 绘制轨迹线
        ax.plot(x, y, z, '-', color=colors[i], linewidth=2, label=f'气泡 {bubble_id}')
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], z[0], color=colors[i], marker='o', s=100)
        ax.scatter(x[-1], y[-1], z[-1], color=colors[i], marker='*', s=100)
    
    ax.set_xlabel('X坐标 (像素)')
    ax.set_ylabel('Y坐标 (像素)')
    ax.set_zlabel('Z坐标 (像素)')
    ax.set_title('气泡运动轨迹', fontsize=16)
    
    # 设置坐标轴范围
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 800)
    ax.set_zlim(0, 800)
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 保存图像
    plt.savefig(os.path.join(output_folder, 'bubble_trajectories_3d.png'), dpi=300)
    plt.close()

# 分析气泡体积随时间的变化
def analyze_bubble_volumes(trajectories):
    """分析气泡体积随时间的变化"""
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    
    for i, (bubble_id, data) in enumerate(trajectories.items()):
        frames = data['frame'].values
        volumes = data['volume(mm^3)'].values
        
        plt.plot(frames, volumes, '-o', color=colors[i], linewidth=2, 
                 label=f'气泡 {bubble_id}')
    
    plt.xlabel('帧号')
    plt.ylabel('体积 (mm³)')
    plt.title('气泡体积随时间的变化', fontsize=16)
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    plt.savefig(os.path.join(output_folder, 'bubble_volumes_over_time.png'), dpi=300)
    plt.close()

# 分析气泡的空间分布
def analyze_spatial_distribution(all_data):
    """分析气泡的空间分布"""
    # 创建3D散点图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 仅使用第一帧的数据进行空间分布分析
    first_frame = all_data[all_data['frame'] == 0]
    
    # 根据气泡体积设置点的大小
    volumes = first_frame['volume(mm^3)'].values
    sizes = volumes / volumes.max() * 200 + 30  # 标准化大小
    
    # 绘制散点图
    scatter = ax.scatter(
        first_frame['x'].values, 
        first_frame['y'].values, 
        first_frame['z'].values, 
        s=sizes, 
        c=first_frame['volume(mm^3)'].values, 
        cmap='viridis', 
        alpha=0.7
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('体积 (mm³)')
    
    ax.set_xlabel('X坐标 (像素)')
    ax.set_ylabel('Y坐标 (像素)')
    ax.set_zlabel('Z坐标 (像素)')
    ax.set_title('第一帧气泡的空间分布', fontsize=16)
    
    # 设置坐标轴范围
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 800)
    ax.set_zlim(0, 800)
    
    # 保存图像
    plt.savefig(os.path.join(output_folder, 'bubble_spatial_distribution.png'), dpi=300)
    plt.close()

# 计算每个气泡的速度
def calculate_bubble_speeds(trajectories):
    """计算气泡的速度"""
    speeds = {}
    
    for bubble_id, data in trajectories.items():
        if len(data) < 2:
            continue
        
        # 计算每一帧之间的速度
        x = data['x'].values
        y = data['y'].values
        z = data['z'].values
        frames = data['frame'].values
        
        # 计算速度
        speeds[bubble_id] = []
        for i in range(1, len(data)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            dz = z[i] - z[i-1]
            
            # 计算欧氏距离
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 假设每帧的时间间隔为1个单位
            speed = distance
            speeds[bubble_id].append(speed)
    
    return speeds

# 分析速度分布
def analyze_speed_distribution(speeds):
    """分析气泡速度的分布"""
    # 将所有气泡的速度合并为一个列表
    all_speeds = []
    for bubble_id, speed_list in speeds.items():
        all_speeds.extend(speed_list)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_speeds, bins=30, alpha=0.7, color='skyblue')
    plt.xlabel('速度 (像素/帧)')
    plt.ylabel('频次')
    plt.title('气泡速度分布', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(os.path.join(output_folder, 'bubble_speed_distribution.png'), dpi=300)
    plt.close()

# 分析气泡大小与速度的关系
def analyze_size_vs_speed(trajectories, speeds):
    """分析气泡大小与速度的关系"""
    # 收集每个气泡的平均体积和平均速度
    avg_volumes = []
    avg_speeds = []
    bubble_ids = []
    
    for bubble_id, data in trajectories.items():
        if bubble_id not in speeds or not speeds[bubble_id]:
            continue
        
        avg_volume = data['volume(mm^3)'].mean()
        avg_speed = np.mean(speeds[bubble_id])
        
        avg_volumes.append(avg_volume)
        avg_speeds.append(avg_speed)
        bubble_ids.append(bubble_id)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_volumes, avg_speeds, alpha=0.7, s=80, c='purple')
    
    # 标记气泡ID
    for i, bubble_id in enumerate(bubble_ids):
        plt.annotate(str(bubble_id), (avg_volumes[i], avg_speeds[i]), 
                     fontsize=9, alpha=0.8)
    
    plt.xlabel('平均体积 (mm³)')
    plt.ylabel('平均速度 (像素/帧)')
    plt.title('气泡体积与速度的关系', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 尝试拟合趋势线
    if len(avg_volumes) > 1:
        try:
            z = np.polyfit(avg_volumes, avg_speeds, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(avg_volumes), max(avg_volumes), 100)
            plt.plot(x_range, p(x_range), 'r--', linewidth=2)
            
            # 添加趋势线方程
            eq_text = f'y = {z[0]:.4f}x + {z[1]:.4f}'
            plt.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        except:
            pass
    
    # 保存图像
    plt.savefig(os.path.join(output_folder, 'bubble_size_vs_speed.png'), dpi=300)
    plt.close()

# 主函数
def main():
    # 加载所有帧的数据
    print("加载所有帧数据...")
    frames_data = load_all_frames()
    if not frames_data:
        print("没有找到有效的帧数据")
        return
    
    print(f"共加载了 {len(frames_data)} 帧数据")
    
    # 分析气泡轨迹
    print("分析气泡轨迹...")
    trajectories, all_data = analyze_bubble_trajectories(frames_data)
    
    # 绘制气泡轨迹
    print("绘制气泡轨迹...")
    plot_bubble_trajectories(trajectories)
    
    # 分析气泡体积随时间的变化
    print("分析气泡体积变化...")
    analyze_bubble_volumes(trajectories)
    
    # 分析气泡的空间分布
    print("分析气泡空间分布...")
    analyze_spatial_distribution(all_data)
    
    # 计算气泡速度
    print("计算气泡速度...")
    speeds = calculate_bubble_speeds(trajectories)
    
    # 分析速度分布
    print("分析速度分布...")
    analyze_speed_distribution(speeds)
    
    # 分析气泡大小与速度的关系
    print("分析气泡大小与速度的关系...")
    analyze_size_vs_speed(trajectories, speeds)
    
    print("分析完成！所有结果已保存到:", output_folder)

if __name__ == "__main__":
    main() 
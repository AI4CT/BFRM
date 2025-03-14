import os
import numpy as np
import pandas as pd
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

# 导入reconstruction3d.py中的函数
from reconstruction3d import (
    visualize_bubbles_3d, 
    visualize_density_estimation, 
    visualize_with_plotly_simple,
    calc_bubble_ellipsoid,
    export_bubble_info,
    find_optimal_y_with_gaussian
)

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
    
    return df_3d

def predict_next_frame(prev_frame_df, current_frame_df):
    """
    基于前一帧的3D信息和当前帧的2D信息，预测当前帧的完整3D信息
    
    参数:
        prev_frame_df: 前一帧的完整3D信息DataFrame
        current_frame_df: 当前帧的2D信息DataFrame
    
    返回:
        包含当前帧预测的3D信息的DataFrame
    """
    pixel_scale = 0.080128  # 像素尺度（毫米/像素）
    
    # 转换当前帧的坐标
    current_frame_3d = convert_coordinates(current_frame_df)
    
    # 创建一个字典，用于存储前一帧中每个气泡的信息
    prev_bubbles = {}
    for _, row in prev_frame_df.iterrows():
        prev_bubbles[row['bubble_id']] = row
    
    # 创建一个列表，用于存储已分配y坐标的气泡信息（用于新气泡的y坐标分配）
    assigned_bubbles = []
    
    # 为当前帧中的每个气泡预测y坐标和其他3D参数
    for idx, row in current_frame_3d.iterrows():
        bubble_id = row['bubble_id']
        
        # 检查该气泡是否在前一帧中存在
        if bubble_id in prev_bubbles:
            prev_bubble = prev_bubbles[bubble_id]
            
            # 1. 预测y坐标 - 添加周期性运动
            # 使用正弦函数模拟之字形运动，周期为10帧
            amplitude = 1.5  # 振幅，大幅减小
            phase_shift = random.uniform(0, 2 * np.pi)  # 随机相位
            
            # 基于前一帧的y坐标，添加周期性变化
            y_change = amplitude * np.sin(phase_shift)
            current_frame_3d.at[idx, 'y'] = prev_bubble['y'] + y_change
            
            # 2. 更新椭球体参数
            # 假设体积保持不变
            volume = prev_bubble['volume(mm^3)']
            current_frame_3d.at[idx, 'volume(mm^3)'] = volume
            
            # 检查气泡是否接近上边界（z坐标接近图像高度）
            z_position = row['z']
            near_top_boundary = z_position > (IMAGE_HEIGHT * 0.85)  # 当z坐标超过图像高度的85%时认为接近上边界
            
            if near_top_boundary:
                # 气泡接近上边界，沿用上一帧的a、b、c值，并添加小的随机变化
                # 但保持体积不变
                prev_a = prev_bubble['a']
                prev_b = prev_bubble['b'] 
                prev_c = prev_bubble['c']
                
                # 添加很小的随机变化（±2%）
                random_factor_a = 1 + random.uniform(-0.02, 0.02)
                random_factor_b = 1 + random.uniform(-0.02, 0.02)
                
                # 计算新的a和b
                new_a = prev_a * random_factor_a
                new_b = prev_b * random_factor_b
                
                # 根据体积守恒计算新的c
                new_c = volume / ((4/3) * np.pi * new_a * new_b) / pixel_scale**3
                
                # 更新当前帧的椭球体参数
                current_frame_3d.at[idx, 'a'] = new_a
                current_frame_3d.at[idx, 'b'] = new_b
                current_frame_3d.at[idx, 'c'] = new_c
            else:
                # 气泡不接近上边界，正常计算c值
                # 椭球体体积公式: V = (4/3) * π * a * b * c
                a = row['a']
                b = row['b']
                # 解出c: c = V / ((4/3) * π * a * b)
                # 体积单位为mm^3，a和b单位为像素，需要考虑像素比例
                # V = (4/3) * π * (a*pixel_scale) * (b*pixel_scale) * (c*pixel_scale)
                # 解出c: c = V / ((4/3) * π * a * b * pixel_scale^2) / pixel_scale
                c = volume / ((4/3) * np.pi * a * b) / pixel_scale**3
                current_frame_3d.at[idx, 'c'] = c
            
            # 3. 预测角度变化
            # 计算角度变化，限制在5度以内
            angle_prev = prev_bubble['angle(degree)']
            angle_curr = row['angle(degree)']
            angle_diff = angle_curr - angle_prev
            
            # 限制角度变化不超过5度
            if abs(angle_diff) > 5:
                if angle_diff > 0:
                    angle_curr = angle_prev + 5
                else:
                    angle_curr = angle_prev - 5
                
                current_frame_3d.at[idx, 'angle(degree)'] = angle_curr
                # 更新弧度值
                current_frame_3d.at[idx, 'angle_rad'] = np.radians(-angle_curr)
            
            # 将已处理的气泡添加到已分配列表中，用于后续新气泡的y坐标分配
            current_bubble = {
                'bubble_id': row['bubble_id'],
                'x': row['x'],
                'y': current_frame_3d.at[idx, 'y'],  # 使用更新后的y坐标
                'z': row['z'],
                'radius': row['radius'],
                'volume(mm^3)': volume,
                'type': row['type'],
                'a': current_frame_3d.at[idx, 'a'],
                'b': current_frame_3d.at[idx, 'b'],
                'c': current_frame_3d.at[idx, 'c'],
                'angle_rad': current_frame_3d.at[idx, 'angle_rad']
            }
            assigned_bubbles.append(current_bubble)
        else:
            # 如果是新出现的气泡，使用与reconstruction3d.py中相同的方法分配y坐标
            # 首先需要计算x坐标的统计特性，用于高斯分布
            x_values = current_frame_3d['x'].values
            # 拟合x坐标的正态分布
            mu_x, std_x = stats.norm.fit(x_values)
            
            # 确保标准差不会太小，以保证充分的空间分布
            std_x = max(std_x, IMAGE_WIDTH / 10)
            
            # 创建当前气泡的信息字典
            current_bubble = {
                'bubble_id': row['bubble_id'],
                'x': row['x'],
                'z': row['z'],
                'radius': row['radius'],
                'volume(mm^3)': row['volume(mm^3)'],
                'type': row['type'],
                'a': row['a'],
                'b': row['b'],
                'c': row['c'],
                'angle_rad': row['angle_rad']
            }
            
            # 使用find_optimal_y_with_gaussian函数分配y坐标
            if assigned_bubbles:
                y_coord = find_optimal_y_with_gaussian(
                    current_bubble,
                    assigned_bubbles,
                    mu_x,
                    std_x
                )
            else:
                # 如果没有已分配的气泡，直接从高斯分布采样
                y_coord = np.random.normal(mu_x, std_x)
                y_coord = max(0, min(y_coord, DEPTH_RANGE))
            
            # 更新气泡的y坐标
            current_frame_3d.at[idx, 'y'] = y_coord
            
            # 添加到已分配列表
            current_bubble['y'] = y_coord
            assigned_bubbles.append(current_bubble)
    
    return current_frame_3d

def save_frame_3d_data(df_3d, output_path, frame_num):
    """
    将气泡三维坐标和椭圆信息保存到以当前帧数命名的CSV文件中
    
    参数:
        df_3d: 包含气泡三维信息的DataFrame
        output_path: 输出路径
        frame_num: 当前帧号
    """
    # 创建专门存放流场三维信息的文件夹
    flow_3d_dir = os.path.join(output_path, "flow_3d_data")
    os.makedirs(flow_3d_dir, exist_ok=True)
    
    # 创建以当前帧数命名的CSV文件路径
    csv_path = os.path.join(flow_3d_dir, f"frame_{frame_num:04d}.csv")
    
    # 保存DataFrame到CSV文件
    df_3d.to_csv(csv_path, index=False)
    print(f"帧 {frame_num} 的3D流场数据已保存到 {csv_path}")
    
    return flow_3d_dir

def process_frame(prev_frame_3d, current_frame_path, output_dir, frame_num):
    """
    处理单个帧：读取2D数据，预测3D位置，保存结果并可视化
    
    参数:
        prev_frame_3d: 前一帧的3D数据
        current_frame_path: 当前帧的2D数据文件路径
        output_dir: 输出目录
        frame_num: 当前帧号
    
    返回:
        当前帧的3D数据，用于下一帧的预测
    """
    print(f"处理第 {frame_num} 帧...")
    
    # 读取当前帧的2D数据
    current_frame = read_bubble_csv(current_frame_path)
    
    # 预测当前帧的3D位置
    current_frame_3d = predict_next_frame(prev_frame_3d, current_frame)

    # 导出气泡详细信息
    bubble_info_dir = export_bubble_info(current_frame_3d, os.path.dirname(output_dir), frame_num)
    print(f"第 {frame_num} 帧的气泡详细信息已导出到: {bubble_info_dir}")
    
    # 创建可视化目录
    visualizations_dir = os.path.join(output_dir, f"frame_{frame_num:04d}")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # 执行可视化
    visualization_methods = {
        "PyVista 3D": lambda: visualize_bubbles_3d(current_frame_3d, visualizations_dir),
        "密度估计": lambda: visualize_density_estimation(current_frame_3d, visualizations_dir),
        "Plotly单一": lambda: visualize_with_plotly_simple(current_frame_3d, visualizations_dir)
    }
    
    # 执行所有可视化方法
    for name, method in visualization_methods.items():
        print(f"执行{name}可视化...")
        try:
            method()
            print(f"{name}可视化完成")
        except Exception as e:
            print(f"{name}可视化失败: {str(e)}")
    
    print(f"第 {frame_num} 帧处理完成，所有可视化文件已保存到: {visualizations_dir}")
    
    return current_frame_3d

def main():
    # 定义输入和输出路径
    bubble_info_dir = r"C:\codebase\BFRM\results\yolo11l-obb\bubble_info"
    bubble_csv_dir = r"C:\codebase\BFRM\results\yolo11l-obb\bubble_csv"
    output_dir = r"C:\codebase\BFRM\results\yolo11l-obb\visualizations_3D"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取第一帧的3D数据（已经处理过的）
    first_frame_path = os.path.join(bubble_info_dir, "frame_0001.csv")
    first_frame_3d = pd.read_csv(first_frame_path)
    
    # 获取所有后续帧的文件路径
    csv_files = [f for f in os.listdir(bubble_csv_dir) if f.endswith('.csv') and f.startswith('frame_')]
    csv_files.sort()  # 确保按帧顺序处理
    
    # 跳过第一帧和第零帧（因为已经有了3D数据）
    if 'frame_0001.csv' in csv_files:
        csv_files.remove('frame_0001.csv')
    if 'frame_0000.csv' in csv_files:
        csv_files.remove('frame_0000.csv')
    
    # 初始化前一帧的3D数据
    prev_frame_3d = first_frame_3d
    
    # 处理每一帧
    for csv_file in tqdm(csv_files, desc="处理帧"):
        # 提取帧号
        frame_num = int(csv_file.split('_')[1].split('.')[0])
        
        # 构建当前帧的文件路径
        current_frame_path = os.path.join(bubble_csv_dir, csv_file)
        
        # 处理当前帧
        current_frame_3d = process_frame(prev_frame_3d, current_frame_path, output_dir, frame_num)
        
        # 更新前一帧的数据
        prev_frame_3d = current_frame_3d
    
    print("所有帧处理完成！")

if __name__ == "__main__":
    main()

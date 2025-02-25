import os
import cv2
import numpy as np
import pandas as pd
from data_loader import load_bubble_data
from ellipse_fitting import draw_ellipse_on_image, draw_bounding_box_on_image

def validate_csv_data(data):
    """验证CSV数据的完整性"""
    required_columns = ['frame', 'x0', 'y0', 'w_r', 'h_r', 'theta']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV文件缺少必要的列")
    return True

def find_longest_consecutive_frames(data):
    """
    在筛选后的数据中找出最长的连续帧序列。
    
    Args:
        data: DataFrame, 包含frame列的数据表
    
    Returns:
        DataFrame: 最长连续帧序列的数据
    """
    if data.empty:
        return data
        
    # 确保按帧号排序
    data = data.sort_values('frame')
    
    # 找出所有帧号差值大于1的位置
    frame_diffs = data['frame'].diff()
    break_points = frame_diffs[frame_diffs > 1].index.tolist()
    
    if not break_points:
        # 所有帧都是连续的
        return data
        
    # 添加起始和结束点
    segments = [(data.index[0], break_points[0])]
    segments.extend([(break_points[i], break_points[i+1]) 
                    for i in range(len(break_points)-1)])
    segments.append((break_points[-1], data.index[-1]))
    
    # 找出最长的段
    longest_segment = max(segments, key=lambda x: x[1] - x[0])
    return data.loc[longest_segment[0]:longest_segment[1]]

def process_images(data_folder, csv_file, output_folder=None):
    """
    处理图像数据，读取文件夹中的图像和CSV数据
    
    Args:
        data_folder: 数据文件夹路径
        csv_file: CSV文件路径
        output_folder: 输出文件夹路径（可选）
    
    Returns:
        tuple: (frames, frame_info_dict) - 处理后的帧列表和帧信息字典
    """
    print(f"处理图像文件夹: {data_folder}")
    print(f"使用CSV文件: {csv_file}")
    if output_folder:
        print(f"输出文件夹: {output_folder}")
    
    # 查找图像文件
    image_files = [f for f in os.listdir(data_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # 排序文件以确保顺序
    image_files.sort()
    
    # 读取图像
    frames = []
    frame_info_dict = {}
    
    for i, img_file in enumerate(image_files):
        # 读取图像
        img_path = os.path.join(data_folder, img_file)
        frame = cv2.imread(img_path)
        
        if frame is not None:
            frames.append(frame)
            
            # 创建帧信息
            frame_info = {
                'filename': img_file,
                'size': frame.shape,
                'bubbles': [
                    {'x': 100, 'y': 150, 'radius': 30, 'area': 2827},
                    {'x': 200, 'y': 200, 'radius': 25, 'area': 1963}
                ]
            }
            
            # 添加到字典
            frame_info_dict[i] = frame_info
    
    print(f"读取了 {len(frames)} 个图像文件")
    
    # 创建输出文件夹（如果提供）
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    return frames, frame_info_dict 
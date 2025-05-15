import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import glob
import re
import math
import shutil

# Set matplotlib font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use DejaVu Sans font
plt.rcParams['axes.unicode_minus'] = False    # Correctly display negative signs

# 气泡体积计算相关常量
PPM = 0.080128  # 像素代表的毫米数（mm/pix）
FLOW_VOLUME = 5.035e-4  # 流场体积（立方米）

class BubbleAnalyzer:
    def __init__(self, csv_dir, output_dir, video_path=None):
        """
        Initialize bubble analyzer
        
        Args:
            csv_dir: Directory containing bubble CSV files
            video_path: Video file path (optional)
        """
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.video_path = video_path
        self.output_dir = os.path.join(output_dir, '2-analysis_results')
        self.bubble_crops_dir = os.path.join(self.output_dir, 'bubble_crops')
        self.visualization_dir = os.path.join(self.output_dir, 'visualizations')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.bubble_crops_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Get all CSV files and sort by frame number
        self.csv_files = self._get_sorted_csv_files()
        self.frame_count = len(self.csv_files)
        
        print(f"Found {self.frame_count} CSV frame files (skipped first frame)")
        
        # If video path is provided, load video
        self.video_frames = None
        if video_path and os.path.exists(video_path):
            self._load_video()
    
    def _get_sorted_csv_files(self):
        """获取并排序CSV文件，跳过第一帧"""
        csv_files = glob.glob(os.path.join(self.csv_dir, "frame_*.csv"))
        
        # Extract frame number and sort
        def get_frame_number(file_path):
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(file_path))
            if match:
                return int(match.group(1))
            return 0
        
        # Sort all CSV files
        csv_files.sort(key=get_frame_number)
        
        # Skip the first CSV file because the first frame is problematic
        if len(csv_files) > 1:
            csv_files = csv_files[1:]
            print("Skipped the first CSV file (first frame)")
        
        return csv_files
    
    def _load_video(self):
        """Load video frames"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {self.video_path}")
                return
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_frames = []
            
            for _ in tqdm(range(frame_count), desc="Loading video frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                self.video_frames.append(frame)
            
            cap.release()
            print(f"Loaded {len(self.video_frames)} video frames")
        except Exception as e:
            print(f"Error loading video: {str(e)}")
    
    def read_frame_data(self, csv_path):
        """
        读取单帧气泡数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            pandas.DataFrame: 气泡数据
        """
        try:
            # 尝试不同编码读取CSV
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
            
            for encoding in encodings:
                try:
                    # 尝试直接读取
                    df = pd.read_csv(csv_path, encoding=encoding)
                    
                    # 检查列名是否可读
                    if not all(isinstance(col, str) for col in df.columns):
                        continue
                    
                    # 预期的列名
                    expected_columns = [
                        'bubble_id', 'x', 'y', 'width', 'height', 
                        'angle(degree)', 'speed(m/s)', 'volume(mm^3)', 
                        'type', 'confidence'
                    ]
                    
                    # 如果列名包含"bubble_id"或类似名称，可能需要重命名
                    if 'bubble_id' in df.columns or any('ID' in col for col in df.columns):
                        # 根据列数决定如何重命名
                        if len(df.columns) == len(expected_columns):
                            df.columns = expected_columns
                        elif len(df.columns) == 9:  # 旧格式，没有confidence列
                            df.columns = expected_columns[:-1]
                            # 添加默认confidence列
                            df['confidence'] = 1.0
                    
                    # 确保必要的列存在
                    missing_columns = [col for col in expected_columns if col not in df.columns]
                    if missing_columns:
                        for col in missing_columns:
                            if col == 'angle(degree)':
                                df[col] = 0.0
                            elif col == 'type':
                                df[col] = 'single'
                            elif col == 'confidence':
                                df[col] = 1.0
                            else:
                                df[col] = 0.0
                    
                    return df
                except:
                    continue
            
            # 如果所有编码尝试都失败，尝试以二进制方式读取并手动解析
            with open(csv_path, 'rb') as f:
                content = f.read()
                
            # 尝试检测列数进行手动解析
            lines = content.split(b'\n')
            if len(lines) > 1:
                # 假设第二行是数据行
                data_line = lines[1]
                fields = data_line.split(b',')
                
                # 根据字段数量决定如何解析
                if len(fields) >= 9:
                    data = []
                    for line in lines[1:]:  # 跳过表头行
                        if line.strip():
                            fields = line.split(b',')
                            if len(fields) >= 9:
                                # 将二进制数据转换为字符串或数字
                                row = []
                                for i, field in enumerate(fields):
                                    if i == 0:  # bubble_id
                                        try:
                                            row.append(int(field))
                                        except:
                                            row.append(0)
                                    elif i == 8:  # type
                                        try:
                                            type_str = field.decode('latin1').strip()
                                            if 'single' in type_str.lower():
                                                row.append('single')
                                            else:
                                                row.append('overlap')
                                        except:
                                            row.append('unknown')
                                    elif i == 9 and len(fields) > 9:  # confidence
                                        try:
                                            row.append(float(field))
                                        except:
                                            row.append(1.0)
                                    else:  # 其他数值字段
                                        try:
                                            row.append(float(field))
                                        except:
                                            row.append(0.0)
                                
                                # 如果没有confidence列，添加默认值
                                if len(row) == 9:
                                    row.append(1.0)
                                
                                data.append(row)
                    
                    # 创建DataFrame
                    df = pd.DataFrame(data, columns=[
                        'bubble_id', 'x', 'y', 'width', 'height', 
                        'angle(degree)', 'speed(m/s)', 'volume(mm^3)', 
                        'type', 'confidence'
                    ])
                    return df
            
            print(f"无法解析文件: {csv_path}")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"读取CSV文件 {csv_path} 时出错: {str(e)}")
            return pd.DataFrame()
    
    def analyze_all_frames(self):
        """Analyze all frame data and calculate statistics"""
        frame_numbers = []
        total_counts = []
        single_counts = []
        overlap_counts = []
        
        for i, csv_file in enumerate(tqdm(self.csv_files, desc="Analyzing frame data")):
            df = self.read_frame_data(csv_file)
            if df.empty:
                continue
                
            # Extract frame number
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
            frame_number = int(match.group(1)) if match else i
            
            # Calculate statistics
            total_count = len(df)
            single_count = len(df[df['type'] == 'single'])
            overlap_count = len(df[df['type'] == 'overlap'])
            
            frame_numbers.append(frame_number)
            total_counts.append(total_count)
            single_counts.append(single_count)
            overlap_counts.append(overlap_count)
        
        # Create statistics DataFrame
        self.stats_df = pd.DataFrame({
            'frame': frame_numbers,
            'total': total_counts,
            'single': single_counts,
            'overlap': overlap_counts
        })
        
        # Check if DataFrame is empty
        if self.stats_df.empty:
            print("Warning: All CSV files read failed, no data to analyze!")
            # Create an empty DataFrame with columns
            self.stats_df = pd.DataFrame(columns=['frame', 'total', 'single', 'overlap'])
            # Add a dummy data point to avoid errors in subsequent processing
            self.stats_df.loc[0] = [0, 0, 0, 0]
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, 'bubble_statistics.csv')
        self.stats_df.to_csv(stats_path, index=False)
        print(f"Statistics data saved to {stats_path}")
        
        return self.stats_df
    
    def create_count_plot(self):
        """Create static plot of bubble counts over frames"""
        if not hasattr(self, 'stats_df'):
            self.analyze_all_frames()
        
        # Check if there's enough data
        if self.stats_df.empty or len(self.stats_df) <= 1 and self.stats_df.iloc[0]['total'] == 0:
            print("Not enough data to create plot")
            return
        
        print("Creating bubble count plot...")
        plt.figure(figsize=(12, 6))
        plt.plot(self.stats_df['frame'], self.stats_df['total'], label='Total Bubbles')
        plt.plot(self.stats_df['frame'], self.stats_df['single'], label='Single Bubbles')
        plt.plot(self.stats_df['frame'], self.stats_df['overlap'], label='Overlapping Bubbles')
        
        plt.xlabel('Frame')
        plt.ylabel('Bubble Count')
        plt.title('Bubble Count Changes Over Frames')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'bubble_count_plot.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Bubble count plot saved to {plot_path}")
    
    def create_count_animation(self):
        """Create animation of bubble counts over frames"""
        if not hasattr(self, 'stats_df'):
            self.analyze_all_frames()
        
        # Check if there's enough data
        if self.stats_df.empty or len(self.stats_df) <= 1 and self.stats_df.iloc[0]['total'] == 0:
            print("Not enough data to create animation")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set initial axis limits
        max_count = max(max(self.stats_df['total'].max(), 1), 100)  # At least 100 to make chart look good
        ax.set_ylim(0, max_count * 1.1)  # Add 10% space at the top
        
        # Initialize lines
        total_line, = ax.plot([], [], label='Total Bubbles', linewidth=2)
        single_line, = ax.plot([], [], label='Single Bubbles', linewidth=2)
        overlap_line, = ax.plot([], [], label='Overlapping Bubbles', linewidth=2)
        
        # Add labels and grid
        ax.set_xlabel('Frame')
        ax.set_ylabel('Bubble Count')
        ax.set_title('Bubble Count Changes Over Frames')
        ax.legend()
        ax.grid(True)
        
        # Update function
        def update(frame):
            # Always show all data up to current frame
            current_data = self.stats_df[self.stats_df['frame'] <= frame]
            
            if len(current_data) > 0:
                x_data = current_data['frame']
                total_line.set_data(x_data, current_data['total'])
                single_line.set_data(x_data, current_data['single'])
                overlap_line.set_data(x_data, current_data['overlap'])
                
                # Adjust X-axis range based on current frame
                if frame <= 50:
                    # For first 50 frames, show fixed range 0-50
                    ax.set_xlim(0, 50)
                else:
                    # After 50 frames, keep starting from 0 but extend to current frame
                    ax.set_xlim(0, frame + 5)  # Add 5 frames of future space
                
                # Dynamically adjust Y-axis to ensure all data is visible
                current_max = max(current_data['total'].max(), 1)
                if current_max > max_count * 0.9:  # If current value approaches 90% of max
                    ax.set_ylim(0, current_max * 1.1)  # Add 10% space
            
            return total_line, single_line, overlap_line
        
        # Create animation
        frames = self.stats_df['frame'].tolist()
        if not frames:
            frames = [0]
        ani = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)
        
        # Save animation
        animation_path = os.path.join(self.output_dir, 'bubble_count_animation.mp4')
        ani.save(animation_path, writer='ffmpeg', fps=10, dpi=200)
        plt.close()
        print(f"Bubble count animation saved to {animation_path}")
    
    def rotate_box_coordinates(self, x, y, width, height, angle_degrees):
        """
        Calculate coordinates for rotated rectangle corners
        
        Args:
            x: Center x coordinate
            y: Center y coordinate
            width: Width
            height: Height
            angle_degrees: Rotation angle in degrees
            
        Returns:
            list: Four corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        
        # Calculate half width and height
        w2 = width / 2
        h2 = height / 2
        
        # Calculate corner coordinates relative to center
        corners_rel = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
        corners_abs = []
        
        # Apply rotation and translate to actual position
        for dx, dy in corners_rel:
            # Rotation transformation
            x_rot = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y_rot = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            
            # Translate to center
            x_abs = x + x_rot
            y_abs = y + y_rot
            
            corners_abs.append((int(x_abs), int(y_abs)))
        
        return corners_abs
    
    def get_rotated_rect_bbox(self, corners):
        """
        Get axis-aligned bounding box for rotated rectangle
        
        Args:
            corners: Four corner coordinates of rotated rectangle
        
        Returns:
            tuple: (x_min, y_min, x_max, y_max)
        """
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return (x_min, y_min, x_max, y_max)
    
    def extract_bubble_regions(self):
        """Extract bubble regions from original video"""
        # Check if video data is available
        if self.video_frames is None:
            print("No video data provided, cannot extract bubble regions")
            return
        
        # Create bubble ID directories (vertical storage)
        bubble_id_dir = os.path.join(self.bubble_crops_dir, 'bubble_id')
        os.makedirs(bubble_id_dir, exist_ok=True)
        frame_id_dir = os.path.join(self.bubble_crops_dir, 'frame_id')
        os.makedirs(frame_id_dir, exist_ok=True)

        # Create subdirectories for each bubble type
        single_dir = os.path.join(self.bubble_crops_dir, 'classification/single')
        overlap_dir = os.path.join(self.bubble_crops_dir, 'classification/overlap')
        os.makedirs(single_dir, exist_ok=True)
        os.makedirs(overlap_dir, exist_ok=True)
        
        # Track all bubble IDs and their information
        all_bubbles = {}  # For storing information of all bubbles, indexed by ID
        
        # First pass: Collect all bubble information
        print("Collecting bubble information...")
        for i, csv_file in enumerate(tqdm(self.csv_files, desc="Collecting bubble information")):
            # Parse frame number
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
            frame_number = int(match.group(1)) if match else i
            
            # Ensure frame number is within valid range
            if frame_number >= len(self.video_frames):
                print(f"Frame number {frame_number} exceeds video range")
                continue
            
            # Read bubble data
            df = self.read_frame_data(csv_file)
            if df.empty:
                continue
                
            # Process each bubble
            for _, bubble in df.iterrows():
                try:
                    bubble_id = int(bubble['bubble_id'])
                    
                    # If this is a new bubble ID, create entry
                    if bubble_id not in all_bubbles:
                        all_bubbles[bubble_id] = []
                    
                    # Add current frame information
                    bubble_info = bubble.to_dict()
                    bubble_info['frame'] = frame_number
                    all_bubbles[bubble_id].append(bubble_info)
                    
                except Exception as e:
                    print(f"Error processing bubble {bubble_id} information: {str(e)}")
        
        # # Second pass: Extract bubble regions by frame (horizontal storage)
        print("Extracting bubble regions (by frame)...")
        for i, csv_file in enumerate(tqdm(self.csv_files, desc="Extracting bubble regions by frame")):
            # Parse frame number
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
            frame_number = int(match.group(1)) if match else i
            
            # Ensure frame number is within valid range
            if frame_number >= len(self.video_frames):
                continue
            
            # Get current frame image
            frame = self.video_frames[frame_number].copy()
            
            # Read bubble data
            df = self.read_frame_data(csv_file)
            if df.empty:
                continue
            
            # Create directory for current frame
            frame_dir = os.path.join(frame_id_dir, f'frame_{frame_number:04d}')
            os.makedirs(frame_dir, exist_ok=True)
            
            # Create bubbles info CSV file for this frame
            frame_csv_path = os.path.join(frame_dir, f'bubbles_info.csv')
            
            # Add frame number column to CSV
            df_with_frame = df.copy()
            df_with_frame['frame'] = frame_number
            df_with_frame.to_csv(frame_csv_path, index=False)
            
            # Process each bubble
            for _, bubble in df.iterrows():
                try:
                    bubble_id = int(bubble['bubble_id'])
                    bubble_type = bubble['type']
                    x, y = float(bubble['x']), float(bubble['y'])
                    width, height = float(bubble['width']), float(bubble['height'])
                    angle = float(bubble['angle(degree)'])
                    
                    # Calculate rotated rectangle corners
                    corners = self.rotate_box_coordinates(x, y, width, height, angle)
                    
                    # 获取轴对齐的边界框
                    x_min, y_min, x_max, y_max = self.get_rotated_rect_bbox(corners)
                    
                    # 获取图像尺寸
                    h, w = frame.shape[:2]
                    
                    # 计算原始截取区域
                    src_x_min = max(0, int(x_min))
                    src_y_min = max(0, int(y_min))
                    src_x_max = min(w, int(x_max))
                    src_y_max = min(h, int(y_max))
                    
                    # 截取原始区域
                    cropped = frame[src_y_min:src_y_max, src_x_min:src_x_max]
                    
                    # 检查是否需要缩放
                    crop_h, crop_w = cropped.shape[:2]
                    max_size = 128
                    scale_factor = 1.0
                    
                    if crop_w > max_size or crop_h > max_size:
                        # 需要缩放，计算缩放比例
                        scale_factor = max_size / max(crop_w, crop_h)
                        new_width = int(crop_w * scale_factor)
                        new_height = int(crop_h * scale_factor)
                        cropped = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # 创建128x128的白色画布
                    bubble_region = np.ones((max_size, max_size, 3), dtype=np.uint8) * 249
                    
                    # 计算居中放置的位置
                    resized_h, resized_w = cropped.shape[:2]
                    dst_x_min = (max_size - resized_w) // 2
                    dst_y_min = (max_size - resized_h) // 2
                    dst_x_max = dst_x_min + resized_w
                    dst_y_max = dst_y_min + resized_h
                    
                    # 确保目标区域在画布范围内
                    dst_x_min = max(0, dst_x_min)
                    dst_y_min = max(0, dst_y_min)
                    dst_x_max = min(max_size, dst_x_max)
                    dst_y_max = min(max_size, dst_y_max)
                    
                    # 调整源区域以匹配目标区域
                    src_h, src_w = cropped.shape[:2]
                    src_x_min = 0
                    src_y_min = 0
                    src_x_max = min(src_w, dst_x_max - dst_x_min)
                    src_y_max = min(src_h, dst_y_max - dst_y_min)
                    
                    # Copy the valid part of the frame to the white canvas
                    if src_x_min < src_x_max and src_y_min < src_y_max:
                        bubble_region[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = cropped[src_y_min:src_y_max, src_x_min:src_x_max]
                        
                        # Save bubble image
                        bubble_image_path = os.path.join(frame_dir, f'{bubble_type}_{bubble_id:04d}.png')
                        cv2.imwrite(bubble_image_path, bubble_region)
                        
                        # Save a copy by type
                        if bubble_type == 'single':
                            target_dir = single_dir
                        else:
                            target_dir = overlap_dir
                        
                        type_image_path = os.path.join(target_dir, f'frame_{frame_number:04d}_bubble_{bubble_id:04d}.png')
                        cv2.imwrite(type_image_path, bubble_region)
                except Exception as e:
                    print(f"Error processing bubble {bubble_id}: {str(e)}")
        
        # Third pass: Extract bubble regions by bubble ID (vertical storage)
        print("Extracting bubble regions (by bubble ID)...")
        for bubble_id, bubble_frames in tqdm(all_bubbles.items(), desc="Extracting regions by bubble ID"):
            # Create directory for this bubble
            bubble_dir = os.path.join(bubble_id_dir, f'bubble_{bubble_id:04d}')
            os.makedirs(bubble_dir, exist_ok=True)
            
            # Create info CSV file for this bubble
            bubble_csv_path = os.path.join(bubble_dir, f'bubble_info.csv')
            
            # Convert bubble information to DataFrame and save
            bubble_df = pd.DataFrame(bubble_frames)
            bubble_df.to_csv(bubble_csv_path, index=False)
            
            # For each frame, extract and save the region of this bubble
            for bubble_info in bubble_frames:
                try:
                    frame_number = bubble_info['frame']
                    
                    # Ensure frame number is within valid range
                    if frame_number >= len(self.video_frames):
                        continue
                        
                    # Get current frame image
                    frame = self.video_frames[frame_number].copy()
                    
                    x, y = float(bubble_info['x']), float(bubble_info['y'])
                    width, height = float(bubble_info['width']), float(bubble_info['height'])
                    angle = float(bubble_info['angle(degree)'])
                    bubble_type = bubble_info['type']
                    
                    # Calculate rotated rectangle corners
                    corners = self.rotate_box_coordinates(x, y, width, height, angle)
                    
                    # Get axis-aligned bounding box
                    x_min, y_min, x_max, y_max = self.get_rotated_rect_bbox(corners)
                    
                    # Calculate the square size (use the larger dimension)
                    square_size = max(x_max - x_min, y_max - y_min)
                    
                    # Calculate new square coordinates centered on the bubble
                    square_x_min = int(x - square_size / 2)
                    square_y_min = int(y - square_size / 2)
                    square_x_max = square_x_min + square_size
                    square_y_max = square_y_min + square_size
                    
                    # Get image dimensions
                    h, w = frame.shape[:2]
                    
                    # 计算原始截取区域
                    src_x_min = max(0, int(x_min))
                    src_y_min = max(0, int(y_min))
                    src_x_max = min(w, int(x_max))
                    src_y_max = min(h, int(y_max))
                    
                    # 截取原始区域
                    cropped = frame[src_y_min:src_y_max, src_x_min:src_x_max]
                    # 检查是否需要缩放
                    crop_h, crop_w = cropped.shape[:2]
                    max_size = 128
                    scale_factor = 1.0
                    
                    if crop_w > max_size or crop_h > max_size:
                        # 需要缩放，计算缩放比例
                        scale_factor = max_size / max(crop_w, crop_h)
                        new_width = int(crop_w * scale_factor)
                        new_height = int(crop_h * scale_factor)
                        cropped = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # 创建128x128的白色画布
                    bubble_region = np.ones((max_size, max_size, 3), dtype=np.uint8) * 249
                    
                    # 计算居中放置的位置
                    resized_h, resized_w = cropped.shape[:2]
                    dst_x_min = (max_size - resized_w) // 2
                    dst_y_min = (max_size - resized_h) // 2
                    dst_x_max = dst_x_min + resized_w
                    dst_y_max = dst_y_min + resized_h
                    
                    # 确保目标区域在画布范围内
                    dst_x_min = max(0, dst_x_min)
                    dst_y_min = max(0, dst_y_min)
                    dst_x_max = min(max_size, dst_x_max)
                    dst_y_max = min(max_size, dst_y_max)
                    
                    # 调整源区域以匹配目标区域
                    src_h, src_w = cropped.shape[:2]
                    src_x_min = 0
                    src_y_min = 0
                    src_x_max = min(src_w, dst_x_max - dst_x_min)
                    src_y_max = min(src_h, dst_y_max - dst_y_min)

                    # Copy the valid part of the frame to the white canvas
                    if src_x_min < src_x_max and src_y_min < src_y_max:
                        bubble_region[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = cropped[src_y_min:src_y_max, src_x_min:src_x_max]
                        
                        # Save bubble image
                        bubble_image_path = os.path.join(bubble_dir, f'frame_{frame_number:04d}.png')
                        cv2.imwrite(bubble_image_path, bubble_region)
                except Exception as e:
                    print(f"Error processing bubble {bubble_id} at frame {frame_number}: {str(e)}")
        
        print(f"Bubble region extraction completed, saved to {self.bubble_crops_dir}")
    
    def draw_rotated_rect(self, frame, x, y, width, height, angle, color=(0, 255, 0), thickness=2):
        """
        Draw rotated rectangle on image
        
        Args:
            frame: Image
            x, y: Rectangle center coordinates
            width, height: Rectangle width and height
            angle: Rotation angle in degrees
            color: BGR color
            thickness: Line thickness
        """
        # Get rotated rectangle corners
        corners = self.rotate_box_coordinates(x, y, width, height, angle)
        
        # Draw rectangle border
        for i in range(4):
            cv2.line(frame, corners[i], corners[(i + 1) % 4], color, thickness)
        
        # Draw center point
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    def visualize_detection_results(self):
        """Visualize bubble detection results on video frames"""
        if self.video_frames is None:
            print("No video data provided, cannot visualize detection results")
            return
        detection_dir = os.path.join(self.visualization_dir, 'detection_results')
        os.makedirs(detection_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(detection_dir, 'detection_results.mp4')
        if len(self.video_frames) > 0:
            first_frame = self.video_frames[0]
            frame_height, frame_width = first_frame.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))
        else:
            print("No available video frames")
            return
        for i, csv_file in enumerate(tqdm(self.csv_files, desc="Visualizing detection results")):
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
            frame_number = int(match.group(1)) if match else i
            if frame_number >= len(self.video_frames):
                print(f"Frame number {frame_number} exceeds video range")
                continue
            frame = self.video_frames[frame_number].copy()
            df = self.read_frame_data(csv_file)
            if df.empty:
                continue
            total_count = len(df)
            single_count = len(df[df['type'] == 'single'])
            overlap_count = len(df[df['type'] == 'overlap'])
            total_volume_m3 = 0.0
            for _, bubble in df.iterrows():
                width = float(bubble['width']) * PPM * 0.001  # m
                height = float(bubble['height']) * PPM * 0.001  # m
                a = width / 2
                b = height / 2
                c = max(width, height) / 2
                bubble_volume = (4/3) * math.pi * a * b * c  # m^3
                total_volume_m3 += bubble_volume
            gas_holdup = total_volume_m3 / FLOW_VOLUME if FLOW_VOLUME > 0 else 0
            gas_holdup_percent = gas_holdup * 100
            for _, bubble in df.iterrows():
                try:
                    bubble_id = int(bubble['bubble_id'])
                    bubble_type = bubble['type']
                    x, y = float(bubble['x']), float(bubble['y'])
                    width, height = float(bubble['width']), float(bubble['height'])
                    angle = float(bubble['angle(degree)'])
                    if bubble_type == 'single':
                        color = (53, 130, 84)  # 深绿色
                        text_fill = (53, 130, 84)
                    else:
                        color = (0, 0, 192)  # 深红色
                        text_fill = (0, 0, 192)
                    self.draw_rotated_rect(frame, x, y, width, height, angle, color=color)
                    # 先画白色粗描边，再画填充色
                    cv2.putText(frame, str(bubble_id), (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, lineType=cv2.LINE_AA)
                    cv2.putText(frame, str(bubble_id), (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_fill, 1, lineType=cv2.LINE_AA)
                except Exception as e:
                    print(f"Error drawing bubble {bubble_id}: {str(e)}")
            # 英文统计信息，分四行，Single和Overlap用对应颜色
            y0 = 30
            dy = 35
            # 先绘制白色描边
            cv2.putText(frame, f"Frame: {frame_number}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Single: {single_count}", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Overlap: {overlap_count}", (20, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Total: {total_count}", (20, y0+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Gas holdup: {gas_holdup_percent:.2f}%", (20, y0+4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            
            # 再绘制原始颜色文字
            cv2.putText(frame, f"Frame: {frame_number}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Single: {single_count}", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (53,130,84), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Overlap: {overlap_count}", (20, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,192), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Total: {total_count}", (20, y0+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Gas holdup: {gas_holdup_percent:.2f}%", (20, y0+4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, lineType=cv2.LINE_AA)
            image_path = os.path.join(detection_dir, f'frame_{frame_number:04d}.png')
            cv2.imwrite(image_path, frame)
            video_writer.write(frame)
        video_writer.release()
        print(f"Detection results video saved to {video_path}")
    
    def visualize_bubble_trajectories(self, max_frames=None):
        """Visualize bubble trajectories
        Args:
            max_frames: Maximum number of history frames to show for each bubble trajectory
        """
        if self.video_frames is None:
            print("No video data provided, cannot visualize bubble trajectories")
            return
        trajectory_dir = os.path.join(self.visualization_dir, 'trajectories')
        os.makedirs(trajectory_dir, exist_ok=True)
        all_frame_numbers = []
        for csv_file in self.csv_files:
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
            if match:
                all_frame_numbers.append(int(match.group(1)))
        if not all_frame_numbers:
            print("No valid frame indices found")
            return
        min_frame = min(all_frame_numbers)
        max_frame = max(all_frame_numbers)
        frames_to_process = self.csv_files
        all_bubbles = {}
        print("Building bubble trajectories...")
        for i, csv_file in enumerate(tqdm(frames_to_process, desc="Building bubble trajectories")):
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
            frame_number = int(match.group(1)) if match else i
            df = self.read_frame_data(csv_file)
            if df.empty:
                continue
            for _, bubble in df.iterrows():
                try:
                    bubble_id = int(bubble['bubble_id'])
                    x, y = float(bubble['x']), float(bubble['y'])
                    width = float(bubble['width'])
                    height = float(bubble['height'])
                    angle = float(bubble['angle(degree)'])
                    bubble_type = bubble['type']
                    if bubble_id not in all_bubbles:
                        all_bubbles[bubble_id] = {
                            'positions': [], 
                            'type': bubble_type,
                            'sizes': []
                        }
                    all_bubbles[bubble_id]['positions'].append((frame_number, x, y))
                    all_bubbles[bubble_id]['sizes'].append((width, height, angle))
                except Exception as e:
                    print(f"Error processing trajectory data: {str(e)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(trajectory_dir, 'bubble_trajectories.mp4')
        if len(self.video_frames) > 0:
            first_frame = self.video_frames[0]
            frame_height, frame_width = first_frame.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))
        else:
            print("No available video frames")
            return
        process_frame_numbers = []
        for csv_file in frames_to_process:
            match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
            if match:
                process_frame_numbers.append(int(match.group(1)))
        process_frame_numbers.sort()
        print("Visualizing bubble trajectories...")
        for frame_number in tqdm(process_frame_numbers, desc="Visualizing bubble trajectories"):
            if frame_number >= len(self.video_frames):
                print(f"Frame number {frame_number} exceeds video range")
                continue
            frame = self.video_frames[frame_number].copy()
            current_csv = None
            for csv_file in frames_to_process:
                match = re.search(r'frame_(\d+)\.csv', os.path.basename(csv_file))
                if match and int(match.group(1)) == frame_number:
                    current_csv = csv_file
                    break
            df = pd.DataFrame()
            if current_csv:
                df = self.read_frame_data(current_csv)
            total_count = len(df)
            single_count = len(df[df['type'] == 'single'])
            overlap_count = len(df[df['type'] == 'overlap'])
            total_volume_m3 = 0.0
            for _, bubble in df.iterrows():
                width = float(bubble['width']) * PPM * 0.001
                height = float(bubble['height']) * PPM * 0.001
                a = width / 2
                b = height / 2
                c = max(width, height) / 2
                bubble_volume = (4/3) * math.pi * a * b * c
                total_volume_m3 += bubble_volume
            gas_holdup = total_volume_m3 / FLOW_VOLUME if FLOW_VOLUME > 0 else 0
            gas_holdup_percent = gas_holdup * 100
            for bubble_id, data in all_bubbles.items():
                positions = data['positions']
                bubble_type = data['type']
                current_positions = [(f, x, y) for f, x, y in positions if f <= frame_number]
                if max_frames is not None and len(current_positions) > max_frames:
                    current_positions = current_positions[-max_frames:]
                if len(current_positions) > 1:
                    if bubble_type == 'single':
                        color = (53, 130, 84)  # 深绿色
                    else:
                        color = (0, 0, 192) # 深红色
                    for j in range(1, len(current_positions)):
                        prev_frame, prev_x, prev_y = current_positions[j-1]
                        curr_frame, curr_x, curr_y = current_positions[j]
                        if curr_frame - prev_frame <= 5:
                            cv2.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), color, 2)
            if not df.empty:
                for _, bubble in df.iterrows():
                    try:
                        bubble_id = int(bubble['bubble_id'])
                        x, y = float(bubble['x']), float(bubble['y'])
                        width, height = float(bubble['width']), float(bubble['height'])
                        angle = float(bubble['angle(degree)'])
                        bubble_type = bubble['type']
                        if bubble_type == 'single':
                            color = (53, 130, 84)  # 深绿色
                            text_fill = (53, 130, 84)
                        else:
                            color = (0, 0, 192)  # 深红色
                            text_fill = (0, 0, 192)
                        self.draw_rotated_rect(frame, x, y, width, height, angle, color=color)
                        # 先画白色粗描边，再画填充色
                        cv2.putText(frame, str(bubble_id), (int(x), int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, lineType=cv2.LINE_AA)
                        cv2.putText(frame, str(bubble_id), (int(x), int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_fill, 1, lineType=cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error drawing bubble {bubble_id} position: {str(e)}")
            y0 = 30
            dy = 35
            # 先绘制白色描边
            cv2.putText(frame, f"Frame: {frame_number}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Single: {single_count}", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Overlap: {overlap_count}", (20, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Total: {total_count}", (20, y0+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Gas holdup: {gas_holdup_percent:.2f}%", (20, y0+4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, lineType=cv2.LINE_AA)
            
            # 再绘制原始颜色文字
            cv2.putText(frame, f"Frame: {frame_number}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Single: {single_count}", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (53,130,84), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Overlap: {overlap_count}", (20, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,192), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Total: {total_count}", (20, y0+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"Gas holdup: {gas_holdup_percent:.2f}%", (20, y0+4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, lineType=cv2.LINE_AA)
            if max_frames is not None:
                cv2.putText(frame, f"Trajectory length: last {max_frames} frames", (20, y0+5*dy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 4, lineType=cv2.LINE_AA)
                cv2.putText(frame, f"Trajectory length: last {max_frames} frames", (20, y0+5*dy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, lineType=cv2.LINE_AA)
            image_path = os.path.join(trajectory_dir, f'frame_{frame_number:04d}.png')
            cv2.imwrite(image_path, frame)
            video_writer.write(frame)
        video_writer.release()
        print(f"Bubble trajectory video saved to {video_path}, trajectories showing last {max_frames} frames")
    
    def summarize_results(self):
        """Output analysis result summary"""
        # Check if there's statistics data
        if hasattr(self, 'stats_df') and not self.stats_df.empty:
            total_frames = len(self.stats_df)
            avg_total = self.stats_df['total'].mean()
            avg_single = self.stats_df['single'].mean()
            avg_overlap = self.stats_df['overlap'].mean()
            max_total = self.stats_df['total'].max()
            
            print("\n=== Analysis Results Summary ===")
            print(f"Total frames processed: {total_frames}")
            print(f"Average total bubbles: {avg_total:.2f}")
            print(f"Average single bubbles: {avg_single:.2f}")
            print(f"Average overlapping bubbles: {avg_overlap:.2f}")
            print(f"Maximum bubble count: {max_total}")
            
            # Output storage location
            print("\n=== Output File Locations ===")
            print(f"Statistics data: {os.path.join(self.output_dir, 'bubble_statistics.csv')}")
            print(f"Static statistics plot: {os.path.join(self.output_dir, 'bubble_count_plot.png')}")
            print(f"Dynamic statistics plot: {os.path.join(self.output_dir, 'bubble_count_animation.mp4')}")
            print(f"Detection results video: {os.path.join(self.visualization_dir, 'detection_results', 'detection_results.mp4')}")
            print(f"Bubble trajectories video: {os.path.join(self.visualization_dir, 'trajectories', 'bubble_trajectories.mp4')}")
            print(f"Bubble region screenshots (horizontal): {self.bubble_crops_dir}")
            print(f"Bubble region screenshots (vertical): {os.path.join(self.bubble_crops_dir, 'by_bubble_id')}")
        else:
            print("No available analysis result statistics data.")

def main():
    """Main function"""
    # Set path
    csv_dir = r"C:\codebase\BFRM\results\yolo11l-obb\1-bubble_csv"
    output_dir = r"C:\codebase\BFRM\results\yolo11l-obb"
    video_path = r"C:\codebase\BFRM\results\yolo11l-obb\0-bubbly_flow.mp4"

    csv_dir = r"C:\codebase\BFRM\results\yolo11m-obb\1-bubble_csv"
    output_dir = r"C:\codebase\BFRM\results\yolo11m-obb"
    video_path = r"C:\codebase\BFRM\results\yolo11l-obb\0-bubbly_flow.mp4"
    
    csv_dir = r"C:\codebase\BFRM\results\RB_bubble_flow1\1-bubble_csv"
    output_dir = r"C:\codebase\BFRM\results\RB_bubble_flow1"
    video_path = r"C:\DataSet_DOOR\Bubbly_flow\RB_bubble_flow1.avi"

    csv_dir = r"C:\codebase\BFRM\results\RB_bubble_flow2\1-bubble_csv"
    output_dir = r"C:\codebase\BFRM\results\RB_bubble_flow2"
    video_path = r"C:\DataSet_DOOR\Bubbly_flow\RB_bubble_flow2.avi"
    
    print("\n===== Starting Bubble Flow Analysis =====\n")
    
    # Create analyzer
    print("Initializing bubble analyzer...")
    analyzer = BubbleAnalyzer(csv_dir, output_dir, video_path)
    
    # Analyze data and create statistics plots
    print("\n===== Step 1: Analyze Data =====")
    analyzer.analyze_all_frames()
    
    print("\n===== Step 2: Create Static Statistics Plot =====")
    analyzer.create_count_plot()
    
    print("\n===== Step 3: Create Dynamic Statistics Plot =====")
    analyzer.create_count_animation()
    
    # Visualize detection results and trajectories
    print("\n===== Step 4: Visualize Detection Results =====")
    analyzer.visualize_detection_results()
    
    print("\n===== Step 5: Visualize Bubble Trajectories =====")
    analyzer.visualize_bubble_trajectories(max_frames=100)  # Only show the trajectory of each bubble for the last 100 frames
    
    # Extract bubble regions
    print("\n===== Step 6: Extract Bubble Regions =====")
    analyzer.extract_bubble_regions()
    
    # Output summary information
    print("\n===== Bubble Flow Analysis Completed =====")
    analyzer.summarize_results()

if __name__ == "__main__":
    main() 
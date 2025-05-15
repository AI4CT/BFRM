#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BFRM Realtime Processor Module
Created by: AI4CT Team
Author: BaodI Yu (yubaodi20@ipe.ac.cn)
"""

import os
import cv2
import numpy as np
import pandas as pd
import time
import logging
import math
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
import pyvista as pl
from scipy import stats
from datetime import datetime
import psutil
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions.predictor import Predictor
import torch
from options import options, update_options
from easydict import EasyDict as edict
import datetime

# 导入YOLO相关模块
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Please install ultralytics package.")

# 修复：导入torch用于判断cuda可用性
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class RealtimeProcessor(QThread):
    """BFRM实时处理核心类"""
    
    # 信号定义
    frame_processed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    processing_finished = pyqtSignal()
    status_updated = pyqtSignal(str)
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 视频相关属性
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_fps = 30.0
        
        # 处理控制属性
        self.is_processing = False
        self.is_paused = False
        self.processing_speed = self.config.default_processing_speed
        self.save_intermediate = self.config.save_intermediate_default
        self.save_stl = self.config.save_stl_default
        self.save_merged_flow_stl = False  # 修改: 不再重复保存STL文件
        
        # YOLO模型相关
        self.model = None
        self.device = 'cpu'
        
        # 气泡追踪相关
        self.trajectories = {}
        self.bubble_colors = {}
        self.frame_info_dict = {}
        self.prev_frame_3d = []
        self.inactive_trajectories = {}
        self.last_seen_frame = {}
        self.trajectory_colors = {}
        
        # 统计信息
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.total_bubble_count = 0
        self.max_bubble_count = 0
        self.bubble_count_history = []
        
        # 输出目录管理 - 统一到session_dir
        self.session_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join('C:/codebase/BFRM/results', self.session_time)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 统一所有子目录到session_dir下
        self.output_dir = self.session_dir  # 修改：output_dir就是session_dir
        self.p2m_temp_dir = os.path.join(self.session_dir, 'p2m_temp')
        self.stl_output_dir = os.path.join(self.session_dir, 'stl_frames')
        self.csv_output_dir = os.path.join(self.session_dir, 'bubble_csv')
        self.detection_frames_dir = os.path.join(self.session_dir, 'detection_frames')
        self.trajectory_data_dir = os.path.join(self.session_dir, 'trajectory_data')
        
        # 创建主要的子目录
        os.makedirs(self.p2m_temp_dir, exist_ok=True)
        os.makedirs(self.stl_output_dir, exist_ok=True)
        
        # 常量定义
        self.IMAGE_WIDTH = self.config.image_width
        self.IMAGE_HEIGHT = self.config.image_height
        self.DEPTH_RANGE = self.config.depth_range
        self.PIXEL_SCALE = self.config.pixel_scale
        
        # Pixel2Mesh相关
        self.p2m_predictor = None
        self.p2m_initialized = False
        self.p2m_device = 'cuda:0' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        self.p2m_options = None
        self.p2m_logger = logging.getLogger('Pixel2Mesh')
        self.p2m_writer = None
        
        # 初始化YOLO模型
        if YOLO_AVAILABLE:
            self.init_yolo_model()
        else:
            self.logger.warning("YOLO不可用，将使用模拟数据")
        
        self.init_pixel2mesh_model()
    
    def init_yolo_model(self):
        """初始化YOLO模型，强制使用GPU（cuda:0）优先"""
        try:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            # 检查设备
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"检测到GPU: {gpu_name}，使用GPU加速")
            else:
                self.device = 'cpu'
                self.logger.info("未检测到可用GPU，使用CPU处理")
            # 检查模型文件是否存在
            if not os.path.exists(self.config.yolo_model_path):
                raise FileNotFoundError(f"YOLO模型文件不存在: {self.config.yolo_model_path}")
            # 加载模型
            self.logger.info(f"正在加载YOLO模型: {self.config.yolo_model_path}")
            self.model = YOLO(self.config.yolo_model_path)
            # 强制将模型转到目标设备
            try:
                self.model.to(self.device)
            except Exception as e:
                self.logger.warning(f"YOLO模型to({self.device})失败: {e}")
            # 预热模型
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False, device=self.device)
            self.logger.info(f"YOLO模型加载成功，使用设备: {self.device}")
            return True
        except Exception as e:
            self.logger.error(f"YOLO模型初始化失败: {str(e)}")
            self.model = None
            return False
    
    def init_pixel2mesh_model(self):
        """初始化Pixel2Mesh模型"""
        try:
            # 加载options
            self.p2m_options = options
            self.p2m_options.num_gpus = 1 if self.p2m_device.startswith('cuda') else 0
            self.p2m_options.pin_memory = False
            self.p2m_options.model.name = 'pixel2mesh'
            # 强制指定权重路径
            self.p2m_options.checkpoint = r'C:/codebase/BFRM/checkpoints/3DBubbles_001190.pt'
            self.p2m_logger.info(f"加载Pixel2Mesh模型，设备: {self.p2m_device}，权重: {self.p2m_options.checkpoint}")
            self.p2m_predictor = Predictor(self.p2m_options, self.p2m_logger, self.p2m_writer)
            self.p2m_predictor.init_fn()
            self.p2m_initialized = True
        except Exception as e:
            self.p2m_logger.error(f"Pixel2Mesh模型初始化失败: {str(e)}")
            self.p2m_predictor = None
            self.p2m_initialized = False
    
    def create_output_directories(self):
        """创建输出目录结构"""
        try:
            # 获取视频名称作为子目录名称
            if self.video_path:
                video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                # 这里原本会创建video_dir，但已删除
                pass
            # 创建中间结果目录（按需）
            if self.save_intermediate:
                os.makedirs(self.csv_output_dir, exist_ok=True)
                os.makedirs(self.detection_frames_dir, exist_ok=True)
                os.makedirs(self.trajectory_data_dir, exist_ok=True)
            self.logger.info(f"创建输出目录: {self.session_dir}")
            return True
        except Exception as e:
            self.logger.error(f"创建输出目录失败: {str(e)}")
            return False
    
    def load_video(self, video_path):
        """加载视频文件"""
        try:
            self.video_path = video_path
            
            # 关闭之前的视频
            if self.cap:
                self.cap.release()
            
            # 打开新视频
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise Exception(f"无法打开视频文件: {video_path}")
            
            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 验证视频信息
            if self.total_frames <= 0:
                raise Exception("视频帧数无效")
            if self.video_fps <= 0:
                self.video_fps = 30.0  # 使用默认帧率
                self.logger.warning("视频帧率无效，使用默认值30 FPS")
            
            self.logger.info(
                f"视频加载成功: {os.path.basename(video_path)}, "
                f"分辨率: {width}×{height}, "
                f"帧数: {self.total_frames}, "
                f"帧率: {self.video_fps:.1f} FPS"
            )
            
            # 创建输出目录
            self.create_output_directories()
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载视频失败: {str(e)}")
            raise e
    
    def start_processing(self):
        """开始处理"""
        if not self.video_path:
            raise Exception("未加载视频文件")
        
        # 重置统计信息
        self.is_processing = True
        self.is_paused = False
        self.current_frame_idx = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.total_bubble_count = 0
        self.max_bubble_count = 0
        self.bubble_count_history = []
        
        # 清空追踪数据
        self.trajectories = {}
        self.bubble_colors = {}
        self.frame_info_dict = {}
        self.prev_frame_3d = []
        self.inactive_trajectories = {}
        self.last_seen_frame = {}
        self.trajectory_colors = {}
        
        # 重置视频到开始位置
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 启动处理线程
        self.start()
        
        self.logger.info("开始实时处理")
    
    def pause_processing(self):
        """暂停处理"""
        self.is_paused = True
        self.logger.info("暂停处理")
    
    def resume_processing(self):
        """恢复处理"""
        self.is_paused = False
        self.logger.info("恢复处理")
    
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        self.is_paused = False
        
        # 等待线程结束
        if self.isRunning():
            self.wait(2000)  # 等待2秒
        
        self.logger.info("停止实时处理")
    
    def reset(self):
        """重置处理器状态"""
        # 停止处理
        self.stop_processing()
        
        # 重置所有状态
        self.current_frame_idx = 0
        self.trajectories = {}
        self.bubble_colors = {}
        self.frame_info_dict = {}
        self.prev_frame_3d = []
        self.inactive_trajectories = {}
        self.last_seen_frame = {}
        self.trajectory_colors = {}
        
        # 重置统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.total_bubble_count = 0
        self.max_bubble_count = 0
        self.bubble_count_history = []
        
        # 重置视频位置
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.logger.info("处理器状态已重置")
    
    def run(self):
        """主处理循环（线程运行方法）"""
        try:
            self.logger.info("处理线程开始运行")
            
            while self.is_processing and self.current_frame_idx < self.total_frames:
                # 检查暂停状态
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning(f"无法读取帧 {self.current_frame_idx}")
                    break
                
                # 处理单帧
                start_time = time.time()
                frame_result = self.process_single_frame(frame, self.current_frame_idx)
                processing_time = time.time() - start_time
                
                # 计算处理FPS
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # 添加FPS和进度信息
                frame_result['fps'] = self.current_fps
                frame_result['progress'] = (self.current_frame_idx + 1) / self.total_frames * 100
                frame_result['processing_time'] = processing_time
                
                # 发送结果
                self.frame_processed.emit(frame_result)
                
                # 控制处理速度
                target_interval = 1.0 / (self.processing_speed * 2)  # 调整处理速度
                sleep_time = max(0, target_interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 更新帧索引
                self.current_frame_idx += 1
                
                # 处理Qt事件
                QApplication.processEvents()
            
            # 处理完成
            if self.is_processing:
                self.logger.info("视频处理完成")
                self.processing_finished.emit()
            
        except Exception as e:
            self.logger.error(f"处理线程异常: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))
        
        finally:
            self.is_processing = False
            self.logger.info("处理线程结束")
    
    def crop_and_pad_bubble_patch(self, frame, center_x, center_y, width, height, angle_deg, out_size=128):
        """裁剪气泡区域并保持原比例填充到指定尺寸"""
        # 计算裁剪区域
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # 确保坐标在图像范围内
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 裁剪原始区域
        crop = frame[y1:y2, x1:x2].copy()
        
        # 如果裁剪区域为空，返回空白图像
        if crop.size == 0:
            return np.zeros((out_size, out_size, 3), dtype=np.uint8)
        
        # 计算缩放比例，保持原比例
        crop_h, crop_w = crop.shape[:2]
        if crop_w > out_size or crop_h > out_size:
            # 如果尺寸超过目标尺寸，等比例缩小
            scale = min(out_size / crop_w, out_size / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标尺寸的空白图像
        result = np.ones((out_size, out_size, 3), dtype=np.uint8) * 253
        
        # 将裁剪的图像放在中心位置
        crop_h, crop_w = crop.shape[:2]
        x_offset = (out_size - crop_w) // 2
        y_offset = (out_size - crop_h) // 2
        result[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = crop
        
        return result

    def process_single_frame(self, frame, frame_idx):
        self.current_frame = frame.copy()
        result = {
            'frame_idx': frame_idx,
            'original_frame': frame.copy(),  # 添加原始帧
            'processed_frame': frame.copy(),
            'bubble_count': 0,
            'single_count': 0,
            'overlap_count': 0,
            'time_seconds': frame_idx / self.video_fps
        }
        try:
            # === 为每帧创建单独的crop文件夹 ===
            frame_crop_dir = os.path.join(self.p2m_temp_dir, f"frame_{frame_idx:04d}")
            os.makedirs(frame_crop_dir, exist_ok=True)
            processed_frame, frame_info = self.process_frame_with_yolo(frame, frame_idx, frame_crop_dir)
            result['processed_frame'] = processed_frame
            result['frame_info'] = frame_info
            bubble_count = len(frame_info)
            single_count = sum(1 for b in frame_info.values() if b.get('type') == 'single')
            overlap_count = sum(1 for b in frame_info.values() if b.get('type') == 'overlap')
            result['bubble_count'] = bubble_count
            result['single_count'] = single_count
            result['overlap_count'] = overlap_count
            self.total_bubble_count += bubble_count
            self.max_bubble_count = max(self.max_bubble_count, bubble_count)
            self.bubble_count_history.append(bubble_count)
            crop_infos = []
            for bubble_id, info in frame_info.items():
                crop_name = f"bubble{bubble_id:04d}.png"
                crop_path = os.path.join(frame_crop_dir, crop_name)
                crop_infos.append((bubble_id, crop_path, info['width'], info['height'], info['angle(degree)']))
            # === Pixel2Mesh批量推理 ===
            mesh_dict = self.run_pixel2mesh_on_crops(crop_infos, frame_crop_dir) if crop_infos else {}
            
            # 用于STL文件保存的数据结构
            bubbles_with_mesh = []
            for bubble_id, info in frame_info.items():
                bubble = {
                    'bubble_id': bubble_id,
                    'x': info.get('x', 0),
                    'y': info.get('y', 0),
                    'width': info.get('width', 0),
                    'height': info.get('height', 0),
                    'angle(degree)': info.get('angle(degree)', 0),
                    'type': info.get('type', 'single'),
                    'confidence': info.get('confidence', 1.0),
                    'z': self.IMAGE_HEIGHT - info.get('y', 0),
                }
                # mesh后处理：只做x方向缩放
                if bubble_id in mesh_dict:
                    mesh_path, width, height, angle = mesh_dict[bubble_id]
                    angle_rad = math.radians(angle)
                    target_x_len = abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad))
                    bubble['mesh_path'] = mesh_path
                    bubble['target_x_len'] = target_x_len
                bubbles_with_mesh.append(bubble)
            
            self.prev_frame_3d = bubbles_with_mesh.copy()
            
            if self.save_intermediate:
                self.save_intermediate_results(frame_info, frame_idx)
            
            # 保存STL文件 (单帧或合并流场)
            if (self.save_stl or self.save_merged_flow_stl) and bubbles_with_mesh:
                self.save_frame_stl_with_mesh(bubbles_with_mesh, frame_idx)
            
            result['log'] = f"处理帧{frame_idx}完成，检测到{bubble_count}个气泡。"
        except Exception as e:
            self.logger.error(f"处理帧 {frame_idx} 失败: {str(e)}")
            result['error'] = str(e)
            result['log'] = f"处理帧{frame_idx}失败: {str(e)}"
        return result

    def process_frame_with_yolo(self, frame, frame_index, frame_crop_dir=None):
        if not self.model:
            return self.simulate_bubble_detection(frame, frame_index)
        processed_frame = frame.copy()
        frame_info = {}
        try:
            device_str = 'cuda:0' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
            results = self.model.track(
                frame,
                persist=True,
                tracker='botsort.yaml',
                verbose=False,
                device=device_str,
                imgsz=640
            )
            current_bubble_ids = set()
            if results and len(results) > 0:
                r = results[0]
                if hasattr(r, 'obb') and r.obb is not None and len(r.obb) > 0:
                    for i in range(len(r.obb)):
                        try:
                            bubble_id = int(r.obb.id[i]) if hasattr(r.obb, 'id') and r.obb.id is not None else i
                            current_bubble_ids.add(bubble_id)
                            if hasattr(r.obb, 'xywhr') and r.obb.xywhr is not None:
                                box_data = r.obb.xywhr[i]
                                center_x, center_y = float(box_data[0]), float(box_data[1])
                                width, height = float(box_data[2]), float(box_data[3])
                                angle = float(box_data[4]) * 180 / np.pi
                            else:
                                center_x = center_y = width = height = angle = 0
                            class_id = int(r.obb.cls[i]) if hasattr(r.obb, 'cls') else 0
                            confidence = float(r.obb.conf[i]) if hasattr(r.obb, 'conf') else 1.0
                            self.update_trajectory(bubble_id, (center_x, center_y), frame_index)
                            speed = self.calculate_bubble_speed(bubble_id)
                            volume = self.calculate_bubble_volume(width, height)
                            bubble_type = 'single' if class_id == 0 else 'overlap'
                            frame_info[bubble_id] = {
                                'id': bubble_id,
                                'x': center_x,
                                'y': center_y,
                                'width': width,
                                'height': height,
                                'angle(degree)': angle,
                                'speed(m/s)': speed,
                                'volume(mm^3)': volume,
                                'class_id': class_id,
                                'type': bubble_type,
                                'confidence': confidence
                            }
                            # 居中裁剪并保存crop到每帧的frame_crop_dir
                            if frame_crop_dir is not None:
                                crop_img = self.crop_and_pad_bubble_patch(frame, center_x, center_y, width, height, angle, out_size=128)
                                crop_name = f"bubble{bubble_id:04d}.png"
                                crop_path = os.path.join(frame_crop_dir, crop_name)
                                cv2.imwrite(crop_path, crop_img)
                                frame_info[bubble_id]['crop_path'] = crop_path
                            else:
                                frame_info[bubble_id]['crop_path'] = None
                        except Exception as e:
                            self.logger.warning(f"处理气泡 {i} 时出错: {str(e)}")
                            continue
            self.update_trajectory_status(current_bubble_ids, frame_index)
            self.draw_bubble_detection(processed_frame, frame_info)
            self.draw_trajectories(processed_frame, frame_index)
            self.add_frame_info_overlay(processed_frame, frame_info, frame_index)
        except Exception as e:
            self.logger.error(f"YOLO处理帧 {frame_index} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, {}
        return processed_frame, frame_info

    def update_trajectory(self, bubble_id, center, frame_index):
        """更新气泡轨迹"""
        if bubble_id not in self.trajectories:
            self.trajectories[bubble_id] = []
        
        # 添加当前位置
        self.trajectories[bubble_id].append((center, frame_index))
        
        # 限制轨迹长度
        max_length = self.config.max_trajectory_length
        if len(self.trajectories[bubble_id]) > max_length:
            self.trajectories[bubble_id].pop(0)
        
        # 更新最后一次出现的帧
        self.last_seen_frame[bubble_id] = frame_index
    
    def calculate_bubble_speed(self, bubble_id):
        """计算气泡速度"""
        if bubble_id not in self.trajectories or len(self.trajectories[bubble_id]) < 2:
            return 0.0
        
        # 取最近两个位置
        pos1, frame1 = self.trajectories[bubble_id][-2]
        pos2, frame2 = self.trajectories[bubble_id][-1]
        
        # 计算距离和时间差
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # 转换为物理距离
        distance_mm = distance_pixels * self.PIXEL_SCALE
        
        # 计算时间差
        frame_diff = frame2 - frame1
        time_diff = frame_diff / self.video_fps
        
        # 计算速度
        if time_diff > 0:
            speed_mm_per_s = distance_mm / time_diff
            speed_m_per_s = speed_mm_per_s / 1000.0
            return speed_m_per_s
        
        return 0.0
    
    def calculate_bubble_volume(self, width, height):
        """计算气泡体积"""
        # 假设气泡为椭球体
        a = width / 2    # X轴半径
        b = height / 2   # Z轴半径
        c = max(width, height) / 2  # Y轴半径（深度方向）
        
        # 椭球体体积公式：V = (4/3) * π * a * b * c
        volume_mm3 = (4/3) * math.pi * a * b * c
        
        return volume_mm3
    
    def update_trajectory_status(self, current_bubble_ids, frame_index):
        """更新轨迹状态，处理消失的气泡"""
        # 检查消失的气泡
        for bubble_id in list(self.trajectories.keys()):
            if bubble_id not in current_bubble_ids:
                # 气泡消失，检查是否超过阈值
                frames_since_last_seen = frame_index - self.last_seen_frame.get(bubble_id, 0)
                if frames_since_last_seen > 10:  # 10帧后移除轨迹
                    # 移动到非活动轨迹
                    self.inactive_trajectories[bubble_id] = self.trajectories.pop(bubble_id)
    
    def draw_bubble_detection(self, frame, bubble_info):
        """在帧上绘制气泡检测结果
        兼容传入单个气泡字典或整个frame_info字典
        """
        # 如果传入的是单个气泡字典，自动转为{bubble_id: info}格式
        if isinstance(bubble_info, dict):
            # 判断是否是单个气泡（含有'x','y','width'等字段且没有'id'键）
            if 'x' in bubble_info and 'y' in bubble_info and 'width' in bubble_info and 'height' in bubble_info and 'id' not in bubble_info:
                bubble_info = {0: bubble_info}
        else:
            # 不是字典，直接返回
            return
        for bubble_id, info in bubble_info.items():
            try:
                cx, cy = info['x'], info['y']
                width, height = info['width'], info['height']
                # 兼容angle和angle(degree)两种写法
                angle = info.get('angle(degree)', 0)
                bubble_type = info.get('type', 'single')
                if bubble_type == 'single':
                    color = (53, 130, 84)
                    text_color = (53, 130, 84)
                else:
                    color = (0, 0, 192)
                    text_color = (0, 0, 192)
                corners = self.get_rotated_rectangle_points(cx, cy, width, height, angle)
                cv2.polylines(frame, [np.array(corners)], True, color, 2)
                cv2.putText(frame, str(bubble_id), (int(cx), int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.putText(frame, str(bubble_id), (int(cx), int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1, lineType=cv2.LINE_AA)
            except Exception as e:
                self.logger.warning(f"绘制气泡 {bubble_id} 时出错: {str(e)}")
    
    def get_rotated_rectangle_points(self, cx, cy, width, height, angle):
        """计算旋转矩形的四个角点"""
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 矩形的四个角点（相对于中心）
        dx = width / 2
        dy = height / 2
        
        # 计算旋转后的角点
        points = []
        for corner_dx, corner_dy in [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]:
            # 旋转变换
            rotated_x = corner_dx * cos_a - corner_dy * sin_a
            rotated_y = corner_dx * sin_a + corner_dy * cos_a
            
            # 平移到实际位置
            x = cx + rotated_x
            y = cy + rotated_y
            points.append((int(x), int(y)))
        
        return points
    
    def draw_trajectories(self, frame, current_frame_index):
        """绘制气泡轨迹"""
        try:
            # 绘制活动轨迹
            for bubble_id, trajectory in self.trajectories.items():
                if len(trajectory) < 2:
                    continue
                
                # 获取或生成轨迹颜色
                if bubble_id not in self.trajectory_colors:
                    self.trajectory_colors[bubble_id] = (
                        np.random.randint(50, 255),
                        np.random.randint(50, 255),
                        np.random.randint(50, 255)
                    )
                
                color = self.trajectory_colors[bubble_id]
                
                # 绘制轨迹线
                for i in range(1, len(trajectory)):
                    pt1 = tuple(map(int, trajectory[i-1][0]))
                    pt2 = tuple(map(int, trajectory[i][0]))
                    
                    # 计算线条粗细（越近越粗）
                    thickness = max(1, 3 - i // 10)
                    cv2.line(frame, pt1, pt2, color, thickness)
            
            # 绘制非活动轨迹（半透明）
            for bubble_id, trajectory in self.inactive_trajectories.items():
                if len(trajectory) < 2:
                    continue
                
                color = self.trajectory_colors.get(bubble_id, (128, 128, 128))
                
                # 绘制半透明轨迹
                for i in range(1, len(trajectory)):
                    pt1 = tuple(map(int, trajectory[i-1][0]))
                    pt2 = tuple(map(int, trajectory[i][0]))
                    cv2.line(frame, pt1, pt2, color, 1)
                    
        except Exception as e:
            self.logger.warning(f"绘制轨迹失败: {str(e)}")
    
    def add_frame_info_overlay(self, frame, frame_info, frame_index):
        """在帧上添加统计信息覆盖层
        
        采用与2-bubble_analysis.py相同的显示格式和颜色
        """
        # 计算气泡数量
        total_count = len(frame_info)
        single_count = sum(1 for info in frame_info.values() if info.get('type') == 'single')
        overlap_count = sum(1 for info in frame_info.values() if info.get('type') == 'overlap')
        
        # 计算总体积和气含率
        PPM = 0.080128  # 像素代表的毫米数（mm/pix）
        FLOW_VOLUME = 5.035e-4  # 流场体积（立方米）
        
        total_volume_m3 = 0.0
        for info in frame_info.values():
            width = info['width'] * PPM * 0.001  # 转换为米
            height = info['height'] * PPM * 0.001  # 转换为米
            a = width / 2
            b = height / 2
            c = max(width, height) / 2
            bubble_volume = (4/3) * math.pi * a * b * c  # 立方米
            total_volume_m3 += bubble_volume
        
        gas_holdup = total_volume_m3 / FLOW_VOLUME if FLOW_VOLUME > 0 else 0
        gas_holdup_percent = gas_holdup * 100
        
        # 绘制统计信息（采用与2-bubble_analysis.py相同的样式）
        y0 = 30
        dy = 35
        
        # 先绘制白色描边
        cv2.putText(frame, f"Frame: {frame_index}", (20, y0), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Single: {single_count}", (20, y0 + dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Overlap: {overlap_count}", (20, y0 + 2*dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Total: {total_count}", (20, y0 + 3*dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Gas holdup: {gas_holdup_percent:.2f}%", (20, y0 + 4*dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, lineType=cv2.LINE_AA)
        
        # 再绘制原始颜色文字
        cv2.putText(frame, f"Frame: {frame_index}", (20, y0), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Single: {single_count}", (20, y0 + dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (53, 130, 84), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Overlap: {overlap_count}", (20, y0 + 2*dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 192), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Total: {total_count}", (20, y0 + 3*dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Gas holdup: {gas_holdup_percent:.2f}%", (20, y0 + 4*dy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    
    def convert_2d_to_3d(self, frame_info, frame_idx):
        """将2D检测结果转换为3D坐标，并用Pixel2Mesh生成mesh"""
        if not frame_info:
            return []
        try:
            bubble_data = []
            for bubble_id, info in frame_info.items():
                bubble_data.append({
                    'bubble_id': bubble_id,
                    'x': info.get('x', 0),
                    'y': info.get('y', 0),
                    'width': info.get('width', 0),
                    'height': info.get('height', 0),
                    'angle(degree)': info.get('angle(degree)', 0),
                    'speed(m/s)': info.get('speed(m/s)', 0),
                    'volume(mm^3)': info.get('volume(mm^3)', 0),
                    'type': info.get('type', 'single'),
                    'confidence': info.get('confidence', 1.0)
                })
            if not bubble_data:
                return []
            df = pd.DataFrame(bubble_data)
            df_converted = self.convert_coordinates(df)
            ellipsoid_params = df_converted.apply(self.calc_bubble_ellipsoid, axis=1)
            df_converted['a'] = ellipsoid_params.apply(lambda x: x['a'])
            df_converted['b'] = ellipsoid_params.apply(lambda x: x['b'])
            df_converted['c'] = ellipsoid_params.apply(lambda x: x['c'])
            df_converted['angle_rad'] = ellipsoid_params.apply(lambda x: x['angle_rad'])
            # 分配或预测Y坐标
            if frame_idx == 0:
                df_3d = self.assign_initial_y_coordinates(df_converted)
            else:
                df_3d = self.predict_next_frame_3d(df_converted, frame_idx)
            bubbles_3d = df_3d.to_dict('records')
            # === Pixel2Mesh三维重建 ===
            for bubble in bubbles_3d:
                try:
                    # 裁剪patch
                    img_patch = self.crop_and_pad_bubble_patch(
                        self.current_frame, bubble['x'], bubble['y'], bubble['width'], bubble['height'], bubble['angle(degree)'], out_size=128
                    )
                    if img_patch.shape[0] < 10 or img_patch.shape[1] < 10:
                        continue
                    mesh_path = self.run_pixel2mesh_on_patch(img_patch, bubble['bubble_id'], frame_idx)
                    if mesh_path:
                        bubble['mesh_path'] = mesh_path
                except Exception as e:
                    self.logger.warning(f"Pixel2Mesh处理气泡{bubble['bubble_id']}失败: {str(e)}")
            self.prev_frame_3d = df_3d.copy()
            return bubbles_3d
        except Exception as e:
            self.logger.error(f"2D转3D失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def convert_coordinates(self, df):
        """转换坐标系并计算椭球体参数"""
        df_3d = df.copy()
        
        # Y轴翻转：图像坐标系Y轴向下，3D坐标系Z轴向上
        df_3d['z'] = self.IMAGE_HEIGHT - df_3d['y']
        
        # 计算椭球体参数
        for idx, row in df_3d.iterrows():
            ellipsoid_params = self.calc_bubble_ellipsoid(row)
            df_3d.at[idx, 'a'] = ellipsoid_params['a']
            df_3d.at[idx, 'b'] = ellipsoid_params['b']
            df_3d.at[idx, 'c'] = ellipsoid_params['c']
            df_3d.at[idx, 'angle_rad'] = ellipsoid_params['angle_rad']
        
        # 计算半径（用于兼容性）
        df_3d['radius'] = df_3d.apply(lambda row: (row['a'] + row['b']) / 2, axis=1)
        
        return df_3d
    
    def calc_bubble_ellipsoid(self, row):
        """计算气泡椭球体参数"""
        width = row['width']
        height = row['height'] 
        angle_deg = -row['angle(degree)']  # 注意符号
        
        # 转换为弧度
        angle_rad = math.radians(angle_deg)
        
        # 椭球体三个半轴
        a = width / 2    # X轴半径
        b = height / 2   # Z轴半径
        c = max(width, height) / 2  # Y轴半径（深度方向）
        
        return {
            'a': a,
            'b': b,
            'c': c,
            'angle_rad': angle_rad,
            'angle_deg': angle_deg
        }
    
    def assign_initial_y_coordinates(self, df_3d):
        """为第一帧分配初始Y坐标（深度），区分气泡类型，模拟locating_bubble31.py的实现"""
        import numpy as np
        from scipy import stats
        
        # 按类型分组
        df_single = df_3d[df_3d['type'] == 'single'].copy()
        df_overlap = df_3d[df_3d['type'] == 'overlap'].copy()
        
        # 计算x坐标的统计特性
        x_values = df_3d['x'].values
        # 拟合x坐标的正态分布
        mu_x, std_x = stats.norm.fit(x_values)
        
        # 确保标准差不会太小，以保证充分的空间分布
        std_x = max(std_x, self.IMAGE_WIDTH / 10)
        
        # 已分配气泡的3D位置
        assigned_bubbles = []
        
        # 为single气泡分配y坐标 - 使用与x相同的高斯分布
        # y坐标表示深度，范围为0到DEPTH_RANGE
        if not df_single.empty:
            self.logger.info("为single气泡分配y坐标(深度)...")
            
            # 按照体积大小排序，大气泡优先
            single_gas = df_single.sort_values('volume(mm^3)', ascending=False)
            
            # 为每个单个气泡分配y坐标
            for idx, row in single_gas.iterrows():
                # 创建包含所有必要参数的字典
                current_bubble = {
                    'bubble_id': row['bubble_id'],
                    'x': row['x'],
                    'z': row['z'],
                    'radius': row['radius'],
                    'volume(mm^3)': row.get('volume(mm^3)', 0),
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
                    y_coord = self.find_optimal_y_with_gaussian(
                        current_bubble, 
                        assigned_bubbles, 
                        mu_x, 
                        std_x
                    )
                else:
                    # 对第一个气泡，直接从高斯分布采样
                    y_coord = np.random.normal(mu_x, std_x)
                    y_coord = max(0, min(y_coord, self.DEPTH_RANGE))
                
                # 更新气泡的y坐标
                df_single.at[idx, 'y'] = y_coord
                
                # 添加到已分配列表
                current_bubble['y'] = y_coord
                assigned_bubbles.append(current_bubble)
        
        # 为overlap气泡分配y坐标
        if not df_overlap.empty:
            self.logger.info("为overlap气泡分配y坐标(深度)...")
            for idx, row in df_overlap.iterrows():
                current_bubble = {
                    'bubble_id': row['bubble_id'],
                    'x': row['x'],
                    'z': row['z'],
                    'radius': row['radius'],
                    'volume(mm^3)': row.get('volume(mm^3)', 0),
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
                y_coord = self.find_optimal_y_with_gaussian(
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
    
    def find_optimal_y_with_gaussian(self, current_bubble, assigned_bubbles, mu, std, max_attempts=50):
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
            y_sample = max(0, min(y_sample, self.DEPTH_RANGE))
            
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
                y_sample = max(0, min(y_sample, self.DEPTH_RANGE))
                
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
    
    def predict_next_frame_3d(self, current_frame_df, frame_idx):
        """基于前一帧预测当前帧的3D坐标"""
        # 创建前一帧气泡ID到信息的映射
        prev_bubbles = {b['bubble_id']: b for b in self.prev_frame_3d}
        
        for idx, row in current_frame_df.iterrows():
            bubble_id = row['bubble_id']
            
            if bubble_id in prev_bubbles:
                # 已存在的气泡，基于前一帧预测
                prev_bubble = prev_bubbles[bubble_id]
                
                # 简单的线性预测加随机扰动
                y_change = np.random.uniform(-2, 2)  # 小幅度随机运动
                
                # 考虑气泡运动趋势（如果有轨迹信息）
                if bubble_id in self.trajectories and len(self.trajectories[bubble_id]) >= 2:
                    # 基于轨迹预测Y方向运动
                    # 这里可以添加更复杂的预测逻辑
                    pass
                
                predicted_y = prev_bubble['y'] + y_change
                # 限制范围
                predicted_y = max(0, min(predicted_y, self.DEPTH_RANGE))
                current_frame_df.at[idx, 'y'] = predicted_y
            else:
                # 新出现的气泡，随机分配
                y_coord = np.random.uniform(self.DEPTH_RANGE * 0.2, self.DEPTH_RANGE * 0.8)
                current_frame_df.at[idx, 'y'] = y_coord
        
        return current_frame_df
    
    def save_intermediate_results(self, frame_info, frame_idx):
        """保存中间处理结果"""
        if not self.save_intermediate or not self.csv_output_dir:
            return
        
        try:
            # 保存气泡检测CSV
            csv_file = os.path.join(self.csv_output_dir, f"frame_{frame_idx:04d}.csv")
            
            if frame_info:
                df = pd.DataFrame.from_dict(frame_info, orient='index')
                # 添加帧索引
                df['frame_idx'] = frame_idx
                df.to_csv(csv_file, index=False)
            
        except Exception as e:
            self.logger.warning(f"保存中间结果失败: {str(e)}")
    
    def save_frame_stl_with_mesh(self, bubbles_3d, frame_idx):
        """保存当前帧的STL文件，优先合并Pixel2Mesh mesh，修正体积和位置"""
        if not bubbles_3d:
            return
        
        # 检查是否需要保存STL文件
        if not self.save_stl:
            return
        try:
            import pyvista as pl
            combined_mesh = None
            for bubble in bubbles_3d:
                mesh = None
                if 'mesh_path' in bubble and os.path.exists(bubble['mesh_path']):
                    try:
                        mesh = pl.read(bubble['mesh_path'])
                        # 体积和位置修正
                        x_len = mesh.bounds[1] - mesh.bounds[0]
                        target_x_len = bubble.get('target_x_len', bubble.get('width', 1))
                        scale_factor = target_x_len / x_len if x_len > 0 else 1.0
                        mesh.translate(-np.array(mesh.center), inplace=True)
                        mesh.scale([scale_factor, scale_factor, scale_factor], inplace=True)
                        mesh.translate([bubble.get('x',0), bubble.get('y',0), bubble.get('z',0)], inplace=True)
                    except Exception as e:
                        self.logger.warning(f"读取Pixel2Mesh mesh失败，跳过: {str(e)}")
                        mesh = None
                if mesh is None:
                    # 回退为椭球体
                    mesh = pl.ParametricEllipsoid(
                        bubble.get('a',1), bubble.get('b',1), bubble.get('c',1)
                    )
                    if bubble.get('angle_rad', 0) != 0:
                        mesh.rotate_z(np.degrees(bubble['angle_rad']))
                    mesh.translate([bubble.get('x',0), bubble.get('y',0), bubble.get('z',0)])
                if combined_mesh is None:
                    combined_mesh = mesh
                else:
                    combined_mesh = combined_mesh.merge(mesh)
            
            # 保存到STL帧目录
            if combined_mesh:
                stl_file = os.path.join(self.stl_output_dir, f"frame_{frame_idx:04d}.stl")
                combined_mesh.save(stl_file)
                self.logger.debug(f"保存STL文件: {stl_file}")
                    
        except Exception as e:
            self.logger.warning(f"保存STL文件失败: {str(e)}")
    
    def get_processing_statistics(self):
        """获取处理统计信息"""
        avg_bubble_count = 0
        if self.bubble_count_history:
            avg_bubble_count = sum(self.bubble_count_history) / len(self.bubble_count_history)
        
        # 获取内存使用情况
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        return {
            'total_frames_processed': self.current_frame_idx,
            'total_frames': self.total_frames,
            'current_fps': self.current_fps,
            'average_bubble_count': avg_bubble_count,
            'max_bubble_count': self.max_bubble_count,
            'memory_usage_mb': memory_mb,
            'active_trajectories': len(self.trajectories),
            'inactive_trajectories': len(self.inactive_trajectories)
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止处理
            self.stop_processing()
            
            # 关闭视频捕获
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # 清理YOLO模型
            if self.model:
                del self.model
                self.model = None
            
            # 清理追踪数据
            self.trajectories.clear()
            self.bubble_colors.clear()
            self.frame_info_dict.clear()
            self.inactive_trajectories.clear()
            self.last_seen_frame.clear()
            self.trajectory_colors.clear()
            
            self.logger.info("处理器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"清理资源时出错: {str(e)}")
    
    def run_pixel2mesh_on_patch(self, crop_path, bubble_id, frame_idx, frame_crop_dir=None):
        """直接读取YOLO检测后保存的crop图片进行Pixel2Mesh推理，obj保存到同一目录"""
        if not self.p2m_initialized or self.p2m_predictor is None:
            return None
        try:
            # crop_path为frame_crop_dir/bubbleXXXX.png
            img = cv2.imread(crop_path)
            if img is None:
                self.logger.warning(f"无法读取crop图片: {crop_path}")
                return None
            if img.shape[:2] != (128, 128):
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            images_tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
            if self.p2m_device.startswith('cuda'):
                images_tensor = images_tensor.cuda()
            # 输出obj路径
            if frame_crop_dir is None:
                frame_crop_dir = os.path.dirname(crop_path)
            meshname = os.path.join(frame_crop_dir, f"bubble{bubble_id:04d}_level2.obj")
            # 构造Pixel2Mesh输入batch，参数名与4.1-bubble_reconstruction.py一致
            input_batch = {
                'images': images_tensor,
                'filepath': [crop_path],
                'outpath': [meshname],
                'images_orig': images_tensor
            }
            with torch.no_grad():
                out = self.p2m_predictor.model(input_batch['images'])
            vert = out["pred_coord"][1][0].cpu().numpy()
            vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
            mesh = np.vstack((vert_v, self.p2m_predictor.ellipsoid.obj_fmt_faces[1]))
            np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")
            return meshname
        except Exception as e:
            self.p2m_logger.error(f"Pixel2Mesh单张推理失败: {str(e)}")
            return None
    
    def run_pixel2mesh_on_crops(self, crop_infos, frame_crop_dir):
        """批量对crop图片运行Pixel2Mesh，返回bubble_id到mesh路径的映射，按帧文件夹读取。参数和格式对齐4.1-bubble_reconstruction.py"""
        if not self.p2m_initialized or self.p2m_predictor is None:
            return {}
        batch_size = 50
        mesh_dict = {}
        for i in range(0, len(crop_infos), batch_size):
            batch = crop_infos[i:i+batch_size]
            images = []
            filepaths = []
            outpaths = []
            for bubble_id, crop_path, width, height, angle in batch:
                img = cv2.imread(crop_path)
                if img is None:
                    self.logger.warning(f"无法读取crop图片: {crop_path}")
                    continue
                if img.shape[:2] != (128, 128):
                    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)
                images.append(img)
                filepaths.append(crop_path)
                # 输出obj路径（与4.1-bubble_reconstruction.py一致）
                outpaths.append(os.path.join(frame_crop_dir, f"bubble{bubble_id:04d}_level2.obj"))
            if not images:
                continue
            images_np = np.stack(images, axis=0)
            images_tensor = torch.from_numpy(images_np).float() / 255.0
            if self.p2m_device.startswith('cuda'):
                images_tensor = images_tensor.cuda()
            # 构造Pixel2Mesh输入batch，参数名与4.1-bubble_reconstruction.py一致
            input_batch = {
                'images': images_tensor,
                'filepath': filepaths,
                'outpath': outpaths,
                'images_orig': images_tensor
            }
            with torch.no_grad():
                out = self.p2m_predictor.model(input_batch['images'])
            for j, (bubble_id, crop_path, width, height, angle) in enumerate(batch):
                if j >= len(images):
                    continue
                vert = out["pred_coord"][1][j].cpu().numpy()
                meshname = outpaths[j]
                vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                mesh = np.vstack((vert_v, self.p2m_predictor.ellipsoid.obj_fmt_faces[1]))
                np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")
                mesh_dict[bubble_id] = (meshname, width, height, angle)
        return mesh_dict

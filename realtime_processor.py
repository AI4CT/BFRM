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
import random

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
        self.save_intermediate = True       # 保存气泡区域截图
        self.save_stl = True  # 保存STL文件
        self.save_merged_flow_stl = False  # 修改: 不再重复保存STL文件
        self.save_usda = True  # 新增: 是否保存USDA文件
        
        # YOLO模型相关
        self.model = None
        self.device = 'cuda:0'
        
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
        self.base_path = os.path.join('C:/codebase/BFRM/results', self.session_time)
        os.makedirs(self.base_path, exist_ok=True)
        
        # 统一所有子目录到base_path下
        self.output_dir = self.base_path  # output_dir就是base_path
        self.p2m_temp_dir = os.path.join(self.base_path, 'p2m_temp')
        self.csv_output_dir = os.path.join(self.base_path, 'bubble_csv')
        
        # 只创建必要的子目录
        os.makedirs(self.p2m_temp_dir, exist_ok=True)
        
        # 常量定义
        self.IMAGE_WIDTH = 1280
        self.IMAGE_HEIGHT = 800
        self.DEPTH_RANGE = 1280
        self.PIXEL_SCALE = 0.080128
        
        # Pixel2Mesh相关
        self.p2m_options = options
        self.p2m_logger = logging.getLogger('Pixel2Mesh')
        self.p2m_writer = None
        self.mesh_level = 2  # 网格精细度，默认使用中等精细度(level 2)
        
        # 初始化Pixel2Mesh参数
        self.p2m_options.num_gpus = 1 if torch.cuda.is_available() else 0
        self.p2m_options.pin_memory = False
        self.p2m_options.model.name = 'pixel2mesh'
        self.p2m_options.dataset.name += '_demo'
        
        # 设置权重路径
        self.p2m_options.checkpoint = os.path.join(self.base_path, 'checkpoints', '3DBubbles_001000.pt')
        if not os.path.exists(self.p2m_options.checkpoint):
            self.p2m_options.checkpoint = 'C:/codebase/BFRM/checkpoints/3DBubbles_001000.pt'
        
        # 初始化YOLO模型
        if YOLO_AVAILABLE:
            self.init_yolo_model()
        else:
            self.logger.warning("YOLO不可用，将使用模拟数据")
        
        # 细分保存选项，可通过config传递
        self.save_crop = getattr(config, 'save_crop', True)  # 是否保存气泡crop
        self.save_obj = getattr(config, 'save_obj', True)    # 是否保存obj
        self.save_stl = getattr(config, 'save_stl', True)    # 是否保存单气泡stl
        self.save_merged_stl = getattr(config, 'save_merged_stl', True)  # 是否保存合并stl
        self.save_crop_render = getattr(config, 'save_crop_render', False)  # 是否保存crop渲染
        self.save_detection_results = getattr(config, 'save_detection_results', False)  # 是否保存检测结果
        self.save_trajectory_results = getattr(config, 'save_trajectory_results', False)  # 是否保存追踪结果
    
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
    
    def create_output_directories(self):
        """创建输出目录结构"""
        try:
            # 获取视频名称作为子目录名称
            if self.video_path:
                video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                # 这里原本会创建video_dir，但已删除
                pass
            # 只创建必要的中间结果目录
            if self.save_intermediate:
                os.makedirs(self.csv_output_dir, exist_ok=True)
            self.logger.info(f"创建输出目录: {self.base_path}")
            return True
        except Exception as e:
            self.logger.error(f"创建输出目录失败: {str(e)}")
            return False
    
    def run_pixel2mesh(self, crop_infos, frame_idx, frame_crop_dir=None, batch_mode=True):
        """使用Predictor标准推理流程，确保与entrypoint_predict.py一致，批量推理所有气泡patch
        
        参数:
            crop_infos: [(bubble_id, crop_image, width, height, angle), ...]
            frame_idx: 当前帧索引
            frame_crop_dir: 输出目录
            batch_mode: 是否使用批处理模式
            
        返回:
            mesh_dict: {bubble_id: (mesh_path, width, height, angle)}
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            import numpy as np
            import os
            from functions.predictor import Predictor
            from options import options, update_options
            
            if frame_crop_dir is None:
                frame_crop_dir = os.path.join(self.p2m_temp_dir, f"frame_{frame_idx:04d}")
            os.makedirs(frame_crop_dir, exist_ok=True)
            
            # 只更新预测文件夹路径
            self.p2m_options.dataset.predict.folder = frame_crop_dir
            
            # 创建预测器并初始化
            predictor = Predictor(self.p2m_options, self.p2m_logger, self.p2m_writer)
            
            # 运行预测
            predictor.predict()
            
            # 收集结果
            mesh_dict = {}
            for bubble_id, _, width, height, angle in crop_infos:
                mesh_path = os.path.join(frame_crop_dir, f"bubble_{bubble_id:04d}.{self.mesh_level}.obj")
                if os.path.exists(mesh_path):
                    mesh_dict[bubble_id] = (mesh_path, width, height, angle)
            
            return mesh_dict
            
        except Exception as e:
            self.logger.error(f"Pixel2Mesh处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def convert_obj_to_stl(self, obj_path, bubble_info, output_dir):
        """将单个气泡的OBJ文件转换为修正后的STL文件，文件名规范为frame_{frame_idx:04d}.stl"""
        try:
            import pyvista as pv
            import numpy as np
            import os
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            if not os.path.isabs(obj_path):
                obj_path = os.path.abspath(obj_path)
            mesh = None
            try:
                mesh = pv.read(obj_path)
                self.logger.debug(f"成功读取OBJ文件(绝对路径): {obj_path}")
            except Exception as e:
                self.logger.warning(f"读取OBJ文件 {obj_path} 失败: {str(e)}")
                return None
            if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
                self.logger.warning(f"OBJ文件 {obj_path} 无效或为空")
                return None
            bubble_id = bubble_info.get('bubble_id', 0)
            x_pos = bubble_info.get('x', 0)
            y_pos = bubble_info.get('y', 0)
            z_pos = bubble_info.get('z', 0) if 'z' in bubble_info else self.IMAGE_HEIGHT - bubble_info.get('y', 0)
            width = bubble_info.get('width', 10)
            height = bubble_info.get('height', 10)
            angle_deg = bubble_info.get('angle(degree)', 0)
            frame_idx = bubble_info.get('frame_idx', 0)
            # 文件名规范
            stl_filename = f"frame_{frame_idx:04d}.stl"
            stl_path = os.path.join(output_dir, stl_filename)
            # ...后续mesh处理和保存不变...
            # 1. 确保mesh是PolyData类型
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            mesh.translate(-np.array(mesh.center), inplace=True)
            angle_rad = np.radians(angle_deg)
            target_x_len = abs(width * np.cos(angle_rad)) + abs(height * np.sin(angle_rad))
            target_z_len = abs(width * np.sin(angle_rad)) + abs(height * np.cos(angle_rad))
            current_x_len = mesh.bounds[1] - mesh.bounds[0]
            current_z_len = mesh.bounds[5] - mesh.bounds[4]
            scale_factor_x = target_x_len / current_x_len if current_x_len > 0 else 1.0
            scale_factor_z = target_z_len / current_z_len if current_z_len > 0 else 1.0
            scale_factor = min(scale_factor_x, scale_factor_z)
            mesh.scale(scale_factor, inplace=True)
            mesh.translate([x_pos, y_pos, z_pos], inplace=True)
            mesh.save(stl_path, binary=True)
            self.logger.debug(f"保存修正后的STL文件: {stl_path}, 点数: {mesh.n_points}, 面数: {mesh.n_cells}")
            return stl_path
        except Exception as e:
            self.logger.error(f"转换OBJ到STL失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def set_mesh_level(self, level):
        """设置Pixel2Mesh网格精细度
        
        参数:
            level: 整数, 1-3之间，分别表示低、中、高三种精细度
        
        返回:
            bool: 设置是否成功
        """
        try:
            level = int(level)
            if level < 1 or level > 3:
                self.logger.warning(f"无效的网格精细度: {level}，必须在1-3之间。使用默认值2。")
                level = 2
            
            self.mesh_level = level
            self.logger.info(f"Pixel2Mesh网格精细度已设置为: level {level} (1=低, 2=中, 3=高)")
            return True
        except Exception as e:
            self.logger.error(f"设置网格精细度失败: {str(e)}")
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
            self.wait(1000)  # 等待2秒
        
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
            
            # 初始化FPS计算相关变量
            self.fps_counter = 0
            self.fps_start_time = time.time()
            self.current_fps = 0.0
            last_fps_update = time.time()
            
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
                
                # 计算处理FPS - 使用滑动窗口平均
                self.fps_counter += 1
                current_time = time.time()
                time_diff = current_time - self.fps_start_time
                
                # 每0.5秒更新一次FPS，使显示更平滑
                if current_time - last_fps_update >= 0.5:
                    self.current_fps = self.fps_counter / time_diff
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                    last_fps_update = current_time
                
                # 添加FPS和进度信息
                frame_result['fps'] = round(self.current_fps, 1)  # 保留一位小数
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

    def process_frame_with_yolo(self, frame, frame_index, frame_crop_dir=None, save_crops=True):
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
                            
                            # 居中裁剪图像 - 只在这里调用一次
                            crop_img = self.crop_and_pad_bubble_patch(frame, center_x, center_y, width, height, angle, out_size=128)
                            
                            # 如果需要保存裁剪图像
                            crop_path = None
                            if save_crops and frame_crop_dir is not None:
                                crop_name = f"bubble_{bubble_id:04d}.png"
                                crop_path = os.path.join(frame_crop_dir, crop_name)
                                cv2.imwrite(crop_path, crop_img)
                            
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
                                'confidence': confidence,
                                'crop_path': crop_path,
                                'crop_image': crop_img  # 保存裁剪后的图像数据
                            }
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

    def process_single_frame(self, frame, frame_idx):
        self.current_frame = frame.copy()
        result = {
            'frame_idx': frame_idx,
            'original_frame': frame.copy(),
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
            
            # 设置是否保存裁剪图像 - 中间结果选项决定
            save_crops = self.save_crop
            
            # 使用YOLO处理帧
            processed_frame, frame_info = self.process_frame_with_yolo(frame, frame_idx, frame_crop_dir, save_crops=save_crops)
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
            
            # 调用convert_2d_to_3d函数为气泡分配三维坐标 - 对每一帧都进行操作
            if bubble_count > 0:
                if frame_idx == 0:
                    self.logger.info("处理第一帧: 为气泡分配初始三维坐标(深度)...")
                else:
                    self.logger.info(f"处理第{frame_idx}帧: 为气泡分配或预测三维坐标(深度)...")
                
                # 准备crop_infos用于Pixel2Mesh处理
                crop_infos = []
                for bubble_id, info in frame_info.items():
                    # 直接使用YOLO检测时保存的裁剪图像
                    crop_image = info.get('crop_image')
                    if crop_image is None:
                        self.logger.warning(f"气泡{bubble_id}没有裁剪图像数据")
                        continue
                    
                    crop_infos.append((
                        bubble_id,
                        crop_image,  # 直接使用内存中的图像数据
                        info['width'],
                        info['height'],
                        info['angle(degree)']
                    ))
                
                # 运行Pixel2Mesh获取mesh_dict
                mesh_dict = self.run_pixel2mesh(crop_infos, frame_idx, frame_crop_dir, batch_mode=True) if crop_infos else {}
                
                # 调用convert_2d_to_3d并传入mesh_dict
                bubbles_3d = self.convert_2d_to_3d(frame_info, frame_idx, mesh_dict)
                
                if bubbles_3d:
                    self.logger.info(f"成功为{len(bubbles_3d)}个气泡分配三维坐标")
                    result['bubbles_3d'] = bubbles_3d
                    
                    # 更新frame_info中的y坐标信息，便于后续帧参考
                    for bubble in bubbles_3d:
                        bubble_id = bubble['bubble_id']
                        if bubble_id in frame_info:
                            frame_info[bubble_id]['y_3d'] = bubble['y']
                    
                    # 确保prev_frame_3d被正确更新，以便下一帧使用
                    self.prev_frame_3d = bubbles_3d.copy()
                    
                    self.logger.info(f"气泡三维坐标分配完成。单个气泡: {single_count}, 重叠气泡: {overlap_count}")
                else:
                    self.logger.warning(f"第{frame_idx}帧三维坐标分配失败")
            
            # 准备带有mesh信息的气泡数据
            bubbles_with_mesh = []
            
            # 如果已经做了三维重建，使用三维坐标
            if 'bubbles_3d' in result and result['bubbles_3d']:
                # 直接使用convert_2d_to_3d函数返回的气泡数据
                bubbles_3d = result['bubbles_3d']
                for bubble in bubbles_3d:
                    bubble_id = bubble['bubble_id']
                    # 添加mesh信息
                    if bubble_id in mesh_dict:
                        mesh_path, width, height, angle = mesh_dict[bubble_id]
                        angle_rad = math.radians(angle)
                        target_x_len = abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad))
                        bubble['mesh_path'] = mesh_path
                        bubble['target_x_len'] = target_x_len
                    bubbles_with_mesh.append(bubble)
                    
                    # 确保y坐标信息被传递到frame_info，用于可视化和其他处理
                    if bubble_id in frame_info:
                        frame_info[bubble_id]['y'] = bubble['y']
                        frame_info[bubble_id]['y_3d'] = bubble['y']
                        frame_info[bubble_id]['has_3d'] = True
                result['bubbles_with_mesh'] = bubbles_with_mesh
            
            if self.save_intermediate:
                self.save_intermediate_results(frame_info, frame_idx)
            
            # 确保result中有bubbles_with_mesh
            if 'bubbles_with_mesh' not in result and 'bubbles_3d' in result:
                result['bubbles_with_mesh'] = result['bubbles_3d']
                
            # 保存STL文件 (单帧或合并流场)
            if (self.save_stl or self.save_merged_stl) and result.get('bubbles_with_mesh'):
                self.save_frame_stl_with_mesh(result['bubbles_with_mesh'], frame_idx)
            
            result['log'] = f"处理帧{frame_idx}完成，检测到{bubble_count}个气泡。"
            
        except Exception as e:
            self.logger.error(f"处理帧 {frame_idx} 失败: {str(e)}")
            result['error'] = str(e)
            result['log'] = f"处理帧{frame_idx}失败: {str(e)}"
        return result

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
    
    def convert_2d_to_3d(self, frame_info, frame_idx, mesh_dict=None):
        """将2D检测结果转换为3D坐标
        
        参数:
            frame_info: 包含气泡信息的字典
            frame_idx: 当前帧索引
            mesh_dict: 可选的mesh信息字典，格式为 {bubble_id: (mesh_path, width, height, angle)}
        
        返回:
            包含3D坐标的气泡列表
        """
        if not frame_info:
            return []
        try:
            # 保存当前帧信息，以便在外部脚本处理结果时使用
            self.current_frame_info = frame_info
            
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
            
            # 创建DataFrame并转换坐标
            df = pd.DataFrame(bubble_data)
            df_converted = self.convert_coordinates(df)
            
            # 计算椭球体参数
            ellipsoid_params = df_converted.apply(self.calc_bubble_ellipsoid, axis=1)
            df_converted['a'] = ellipsoid_params.apply(lambda x: x['a'])
            df_converted['b'] = ellipsoid_params.apply(lambda x: x['b'])
            df_converted['c'] = ellipsoid_params.apply(lambda x: x['c'])
            df_converted['angle_rad'] = ellipsoid_params.apply(lambda x: x['angle_rad'])
            
            # 分配或预测Y坐标
            if frame_idx == 0:
                self.logger.info("第一帧：分配初始三维坐标...")
                df_3d = self.assign_initial_y_coordinates(df_converted)
            else:
                self.logger.info(f"第{frame_idx}帧：预测三维坐标...")
                df_3d = self.predict_next_frame_3d(df_converted, frame_idx)
            
            # 转换为字典列表
            bubbles_3d = df_3d.to_dict('records')
            self.logger.info(f"共转换{len(bubbles_3d)}个气泡到3D空间")
            
            # 更新气泡数据，添加mesh信息
            for bubble in bubbles_3d:
                bubble_id = bubble['bubble_id']
                if mesh_dict and bubble_id in mesh_dict:
                    mesh_path, width, height, angle = mesh_dict[bubble_id]
                    bubble['mesh_path'] = mesh_path
                    # 同时保存原始椭球体参数供后续处理
                    bubble['original_a'] = bubble['a']
                    bubble['original_b'] = bubble['b']
                    bubble['original_c'] = bubble['c']
                    # 计算目标x长度用于缩放
                    angle_rad = math.radians(angle)
                    target_x_len = abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad))
                    bubble['target_x_len'] = target_x_len
            
            # 更新self.prev_frame_3d用于下一帧的预测
            self.prev_frame_3d = bubbles_3d.copy()
            
            # 返回包含三维坐标和mesh信息的气泡列表
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
        
        # 对y方向使用与x方向相近的标准差
        # 允许有10%的浮动范围
        std_y = std_x * np.random.uniform(0.9, 1.1)
        
        self.logger.info(f"气泡x坐标分布：均值={mu_x:.2f}，标准差={std_x:.2f}")
        self.logger.info(f"气泡y坐标分布：使用均值={mu_x:.2f}，标准差={std_y:.2f}")
        self.logger.info(f"初始坐标分配：共有{len(df_overlap)}个overlap气泡和{len(df_single)}个single气泡")
        
        # 已分配气泡的3D位置
        assigned_bubbles = []
        
        # 先为overlap气泡分配y坐标，因为它们重叠概率更高
        if not df_overlap.empty:
            self.logger.info("先为overlap气泡分配y坐标(深度)...")
            # 按照体积大小排序，大气泡优先
            overlap_gas = df_overlap.sort_values('volume(mm^3)', ascending=False)
            
            for idx, row in overlap_gas.iterrows():
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
                # 使用与x相同的高斯分布均值，标准差与x相似
                if len(assigned_bubbles) > 0:
                    y_coord = self.find_optimal_y_with_gaussian(
                        current_bubble, 
                        assigned_bubbles, 
                        mu_x, 
                        std_y  # 使用与x相似的标准差
                    )
                else:
                    # 对第一个气泡，直接从高斯分布采样
                    y_coord = np.random.normal(mu_x, std_y)
                    y_coord = max(0, min(y_coord, self.DEPTH_RANGE))
                
                # 更新气泡的y坐标
                df_overlap.at[idx, 'y'] = y_coord
                
                # 添加到已分配列表
                current_bubble['y'] = y_coord
                assigned_bubbles.append(current_bubble)
                
                # 记录每个气泡的分配信息
                self.logger.debug(f"气泡ID {row['bubble_id']} (overlap, 体积={row.get('volume(mm^3)', 0):.2f}): 分配y坐标={y_coord:.2f}")
        
        # 然后为single气泡分配y坐标
        if not df_single.empty:
            self.logger.info("然后为single气泡分配y坐标(深度)...")
            
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
                        std_y  # 使用与x相似的标准差
                    )
                else:
                    # 对第一个气泡，直接从高斯分布采样
                    y_coord = np.random.normal(mu_x, std_y)
                    y_coord = max(0, min(y_coord, self.DEPTH_RANGE))
                
                # 更新气泡的y坐标
                df_single.at[idx, 'y'] = y_coord
                
                # 添加到已分配列表
                current_bubble['y'] = y_coord
                assigned_bubbles.append(current_bubble)
                
                # 记录每个气泡的分配信息
                self.logger.debug(f"气泡ID {row['bubble_id']} (single, 体积={row.get('volume(mm^3)', 0):.2f}): 分配y坐标={y_coord:.2f}")
        
        # 合并overlap和single气泡
        df_combined = pd.concat([df_overlap, df_single], ignore_index=True)
        
        # === 评估y坐标分布并调整 ===
        if len(df_combined) > 3:  # 至少需要几个气泡才能评估分布
            # 计算分配后的y坐标统计特性
            y_values = df_combined['y'].values
            mu_y, std_y_actual = stats.norm.fit(y_values)
            
            # 比较y和x的标准差，检查是否在允许的浮动范围内
            std_ratio = std_y_actual / std_x
            self.logger.info(f"坐标分布评估: x标准差={std_x:.2f}, y标准差={std_y_actual:.2f}, 比值={std_ratio:.2f}")
            
            # 如果y方向标准差超出允许的浮动范围（0.9-1.1），则调整
            if std_ratio < 0.9 or std_ratio > 1.1:
                self.logger.info(f"y方向分布需要调整（标准差比值={std_ratio:.2f}）")
                
                # 计算调整因子，使y的标准差接近x的标准差
                target_std_y = std_x
                adjust_factor = target_std_y / std_y_actual
                
                # 调整y坐标，扩展或收缩分布
                # 保持平均值不变，按比例调整与平均值的距离
                for idx in df_combined.index:
                    y_current = df_combined.at[idx, 'y']
                    y_adjusted = mu_y + (y_current - mu_y) * adjust_factor
                    y_adjusted = max(0, min(y_adjusted, self.DEPTH_RANGE))
                    df_combined.at[idx, 'y'] = y_adjusted
                
                # 检查调整后的分布
                y_values_adjusted = df_combined['y'].values
                mu_y_adj, std_y_adj = stats.norm.fit(y_values_adjusted)
                self.logger.info(f"调整后: y均值={mu_y_adj:.2f}, 标准差={std_y_adj:.2f}, 比值={std_y_adj/std_x:.2f}")
        
        self.logger.info(f"初始y坐标分配完成: 总共为{len(df_combined)}个气泡分配了三维坐标")
        
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
        # 定义初始距离阈值 - 基于气泡半径和类型
        radius = current_bubble['radius']
        bubble_type = current_bubble.get('type', 'single')
        
        # 对于重叠气泡，使用更大的阈值，因为它们更可能在空间中靠近其他气泡
        type_factor = 1.0 if bubble_type == 'single' else 1.2
        
        # 体积越大，要求的距离阈值越大
        volume_factor = 1.0
        volume = current_bubble.get('volume(mm^3)', 0)
        if volume > 0:
            # 将体积因子限制在合理范围内
            volume_factor = min(1.5, max(1.0, (volume / 1000) * 0.1 + 1.0))
        
        # 综合考虑半径、类型和体积
        initial_threshold = radius * 2.5 * type_factor * volume_factor
        
        # 最小可接受阈值 - 不允许气泡完全重叠
        min_threshold = radius * 1.2
        
        # 当前阈值
        threshold = initial_threshold
        
        best_y = None
        min_overlap = float('inf')
        last_attempt_y = None  # 记录最后一次尝试的y坐标
        
        # 使用与输入标准差近似的值
        initial_std = std * np.random.uniform(0.95, 1.05)  # 允许5%的随机波动
        
        # 尝试多次采样
        for attempt in range(max_attempts * 2):  # 增加尝试次数
            # 从高斯分布采样
            # 只有在多次尝试失败后才逐渐增加或减小标准差
            if attempt < max_attempts:
                current_std = initial_std  # 前半部分尝试保持初始标准差
            else:
                # 后半部分尝试调整标准差，但仍保持在合理范围内
                if attempt % 2 == 0:  # 偶数次尝试增加标准差
                    current_std = initial_std * (1.0 + (attempt - max_attempts) / max_attempts * 0.2)
                else:  # 奇数次尝试减小标准差
                    current_std = initial_std * (1.0 - (attempt - max_attempts) / max_attempts * 0.1)
            
            # 如果有前一次尝试的位置且不成功，尝试远离该位置
            if last_attempt_y is not None and attempt > 10:  # 前10次保持随机性
                # 以前一次尝试为基础，选择远离的方向
                if np.random.random() > 0.5:
                    # 向正方向偏移
                    y_sample = last_attempt_y + np.random.normal(0, current_std * 0.3)
                else:
                    # 向负方向偏移
                    y_sample = last_attempt_y - np.random.normal(0, current_std * 0.3)
            else:
                # 直接从高斯分布采样，使用固定的均值
                y_sample = np.random.normal(mu, current_std)
            
            # 确保y坐标在有效范围内
            y_sample = max(0, min(y_sample, self.DEPTH_RANGE))
            last_attempt_y = y_sample  # 记录本次尝试的y坐标
            
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
                
                # 考虑两个气泡的类型 - 如果两个都是overlap，应该有更大的距离阈值
                bubble_interaction_factor = 1.0
                if current_bubble.get('type', 'single') == 'overlap' and bubble.get('type', 'single') == 'overlap':
                    bubble_interaction_factor = 1.3  # overlap-overlap交互需要更大距离
                
                adjusted_threshold = threshold * bubble_interaction_factor
                
                # 计算重叠程度 - 如果距离小于半径和加上阈值，则视为有潜在重叠
                if distance < sum_of_radii + adjusted_threshold:
                    current_overlap += sum_of_radii + adjusted_threshold - distance
                    all_distances_ok = False
            
            # 如果所有距离都满足阈值，直接返回
            if all_distances_ok:
                self.logger.debug(f"为气泡找到理想y坐标: {y_sample:.2f}，尝试次数: {attempt+1}/{max_attempts*2}，使用标准差: {current_std:.2f}")
                return y_sample
            
            # 更新最佳结果
            if current_overlap < min_overlap:
                min_overlap = current_overlap
                best_y = y_sample
        
        # 如果找不到满足条件的，降低阈值再试一次
        if threshold > min_threshold:
            self.logger.debug(f"降低阈值尝试分配y坐标, 从 {threshold:.2f} 降至 {max(min_threshold, threshold * 0.8):.2f}")
            threshold = max(min_threshold, threshold * 0.8)
            
            # 额外尝试
            for attempt in range(max_attempts):
                # 从高斯分布采样
                y_sample = np.random.normal(mu, initial_std * 0.9)
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
                    sum_of_radii = radius + bubble['radius']
                    if distance < sum_of_radii + threshold:
                        current_overlap += sum_of_radii + threshold - distance
                        all_distances_ok = False
                
                # 如果所有距离都满足阈值，直接返回
                if all_distances_ok:
                    self.logger.debug(f"使用降低的阈值为气泡找到合适的y坐标: {y_sample:.2f}")
                    return y_sample
                
                # 更新最佳结果
                if current_overlap < min_overlap:
                    min_overlap = current_overlap
                    best_y = y_sample
        
        # 如果仍然找不到理想的位置，返回最佳结果或随机生成
        if best_y is not None:
            self.logger.debug(f"未找到理想y坐标，使用次优解: {best_y:.2f}，重叠度: {min_overlap:.2f}")
            return best_y
        else:
            # 生成随机值，使用与原始标准差接近的值
            random_y = np.random.normal(mu, initial_std)
            random_y = max(0, min(random_y, self.DEPTH_RANGE))
            self.logger.debug(f"无法找到任何合适的y坐标，使用随机值: {random_y:.2f}")
            return random_y
    
    def predict_next_frame_3d(self, current_frame_df, frame_idx):
        """基于前一帧预测当前帧的3D坐标，参考3.2-reconstructing_bubble.py的实现"""
        import numpy as np
        import random
        from scipy import stats
        
        # 确认是否有前一帧的数据
        if not self.prev_frame_3d:
            self.logger.warning(f"第{frame_idx}帧：没有前一帧的3D数据，将使用初始Y坐标分配")
            return self.assign_initial_y_coordinates(current_frame_df)
        
        # 创建前一帧气泡ID到信息的映射
        prev_bubbles = {b['bubble_id']: b for b in self.prev_frame_3d}
        
        # 创建已分配气泡列表，用于新气泡的y坐标分配
        assigned_bubbles = []
        
        # 计算x坐标的统计特性，用于高斯分布
        x_values = current_frame_df['x'].values
        mu_x, std_x = stats.norm.fit(x_values)
        std_x = max(std_x, self.IMAGE_WIDTH / 10)  # 确保标准差不会太小
        
        # 对y方向使用与x方向相近的标准差，允许10%的浮动
        std_y = std_x * np.random.uniform(0.9, 1.1)
        
        # 像素比例 - 用于体积计算
        pixel_scale = self.PIXEL_SCALE
        
        # 先为现有气泡分配y坐标（从前一帧继承）
        existing_rows = []
        new_rows = []
        
        # 将气泡分成现有气泡和新气泡两组
        for idx, row in current_frame_df.iterrows():
            bubble_id = row['bubble_id']
            if bubble_id in prev_bubbles:
                existing_rows.append((idx, row, bubble_id))
            else:
                new_rows.append((idx, row, bubble_id))
        
        # 处理现有气泡 - 这些气泡在前一帧中已经存在
        self.logger.debug(f"第{frame_idx}帧: 处理{len(existing_rows)}个现有气泡")
        for idx, row, bubble_id in existing_rows:
            # 已存在的气泡，基于前一帧预测
            prev_bubble = prev_bubbles[bubble_id]
            
            # 1. 预测y坐标 - 添加轻微随机变化
            # 使用小的随机变化代替周期性运动，保持与前一帧的连续性
            random_change = random.uniform(-1.5, 1.5)  # 允许正负1.5像素的随机变化
            
            # 基于前一帧的y坐标，添加小的随机变化
            predicted_y = prev_bubble.get('y', mu_x) + random_change
            
            # 考虑气泡运动趋势
            if bubble_id in self.trajectories and len(self.trajectories[bubble_id]) >= 2:
                # 可以添加更复杂的预测逻辑，如运动方向和速度
                pass
            
            # 限制范围
            predicted_y = max(0, min(predicted_y, self.DEPTH_RANGE))
            current_frame_df.at[idx, 'y'] = predicted_y
            
            # 2. 更新椭球体参数（可选）
            # 优先使用前一帧的体积信息，如果没有则计算当前帧的体积
            volume = prev_bubble.get('volume(mm^3)', 0)
            if volume <= 0:
                # 如果前一帧没有体积信息，计算当前帧的体积
                a = row['a']
                b = row['b']
                c = max(a, b)  # 默认深度
                volume = (4/3) * np.pi * a * b * c * (pixel_scale**3)
            
            # 检查气泡是否接近上边界（Z坐标接近图像高度）
            z_position = row['z']
            near_top_boundary = z_position > (self.IMAGE_HEIGHT * 0.85)
            
            if near_top_boundary:
                # 气泡接近上边界，使用微小变化
                prev_a = prev_bubble.get('a', row['a'])
                prev_b = prev_bubble.get('b', row['b'])
                prev_c = prev_bubble.get('c', max(row['width'], row['height']) / 2)
                
                # 添加小的随机变化（±2%）
                random_factor_a = 1 + random.uniform(-0.02, 0.02)
                random_factor_b = 1 + random.uniform(-0.02, 0.02)
                
                # 计算新的a和b
                new_a = prev_a * random_factor_a
                new_b = prev_b * random_factor_b
                
                # 根据体积守恒计算新的c（考虑像素比例）
                new_c = volume / ((4/3) * np.pi * new_a * new_b) / (pixel_scale**3)
                
                # 更新当前帧的椭球体参数
                current_frame_df.at[idx, 'a'] = new_a
                current_frame_df.at[idx, 'b'] = new_b
                current_frame_df.at[idx, 'c'] = new_c
            else:
                # 气泡不接近上边界，保持a和b不变，根据体积计算c
                a = row['a']
                b = row['b']
                c = volume / ((4/3) * np.pi * a * b) / (pixel_scale**3)
                current_frame_df.at[idx, 'c'] = c
            
            # 3. 预测角度变化（轻微限制角度变化）
            angle_prev = prev_bubble.get('angle(degree)', 0)
            angle_curr = row['angle(degree)']
            angle_diff = angle_curr - angle_prev
            
            # 限制角度变化不超过5度
            if abs(angle_diff) > 5:
                if angle_diff > 0:
                    angle_curr = angle_prev + 5
                else:
                    angle_curr = angle_prev - 5
                
                current_frame_df.at[idx, 'angle(degree)'] = angle_curr
                # 更新弧度值
                current_frame_df.at[idx, 'angle_rad'] = np.radians(-angle_curr)
            
            # 将处理后的气泡添加到已分配列表
            current_bubble = {
                'bubble_id': row['bubble_id'],
                'x': row['x'],
                'y': current_frame_df.at[idx, 'y'],  # 使用更新后的值
                'z': row['z'],
                'radius': row['radius'],
                'type': row['type'],
                'a': current_frame_df.at[idx, 'a'],
                'b': current_frame_df.at[idx, 'b'],
                'c': current_frame_df.at[idx, 'c'],
                'angle_rad': current_frame_df.at[idx, 'angle_rad']
            }
            assigned_bubbles.append(current_bubble)
        
        # 处理新出现的气泡，先处理overlap类型，再处理single类型
        # 按照类型分组
        overlap_new_rows = [(idx, row, bid) for idx, row, bid in new_rows if row['type'] == 'overlap']
        single_new_rows = [(idx, row, bid) for idx, row, bid in new_rows if row['type'] == 'single']
        
        # 将overlap气泡按体积降序排序
        overlap_new_rows.sort(key=lambda x: x[1].get('volume(mm^3)', 0), reverse=True)
        # 将single气泡按体积降序排序
        single_new_rows.sort(key=lambda x: x[1].get('volume(mm^3)', 0), reverse=True)
        
        # 先处理overlap气泡
        if overlap_new_rows:
            self.logger.debug(f"第{frame_idx}帧: 处理{len(overlap_new_rows)}个新的overlap气泡")
            
            for idx, row, bubble_id in overlap_new_rows:
                current_bubble = {
                    'bubble_id': bubble_id,
                    'x': row['x'],
                    'z': row['z'],
                    'radius': row['radius'],
                    'type': row['type']
                }
                
                # 添加椭球体参数
                if 'a' in row and 'b' in row and 'c' in row and 'angle_rad' in row:
                    current_bubble['a'] = row['a']
                    current_bubble['b'] = row['b']
                    current_bubble['c'] = row['c']
                    current_bubble['angle_rad'] = row['angle_rad']
                
                # 使用find_optimal_y_with_gaussian函数分配y坐标
                if assigned_bubbles:
                    y_coord = self.find_optimal_y_with_gaussian(
                        current_bubble,
                        assigned_bubbles,
                        mu_x,
                        std_y  # 使用与x相近的标准差
                    )
                else:
                    # 如果没有已分配的气泡，直接从高斯分布采样
                    y_coord = np.random.normal(mu_x, std_y)
                    y_coord = max(0, min(y_coord, self.DEPTH_RANGE))
                
                # 更新气泡的y坐标
                current_frame_df.at[idx, 'y'] = y_coord
                
                # 添加到已分配列表
                current_bubble['y'] = y_coord
                assigned_bubbles.append(current_bubble)
            
        # 然后处理single气泡
        if single_new_rows:
            self.logger.debug(f"第{frame_idx}帧: 处理{len(single_new_rows)}个新的single气泡")
            
            for idx, row, bubble_id in single_new_rows:
                current_bubble = {
                    'bubble_id': bubble_id,
                    'x': row['x'],
                    'z': row['z'],
                    'radius': row['radius'],
                    'type': row['type']
                }
                
                # 添加椭球体参数
                if 'a' in row and 'b' in row and 'c' in row and 'angle_rad' in row:
                    current_bubble['a'] = row['a']
                    current_bubble['b'] = row['b']
                    current_bubble['c'] = row['c']
                    current_bubble['angle_rad'] = row['angle_rad']
                
                # 使用find_optimal_y_with_gaussian函数分配y坐标
                if assigned_bubbles:
                    y_coord = self.find_optimal_y_with_gaussian(
                        current_bubble,
                        assigned_bubbles,
                        mu_x,
                        std_y  # 使用与x相近的标准差
                    )
                else:
                    # 如果没有已分配的气泡，直接从高斯分布采样
                    y_coord = np.random.normal(mu_x, std_y)
                    y_coord = max(0, min(y_coord, self.DEPTH_RANGE))
                
                # 更新气泡的y坐标
                current_frame_df.at[idx, 'y'] = y_coord
                
                # 添加到已分配列表
                current_bubble['y'] = y_coord
                assigned_bubbles.append(current_bubble)
                
        total_new_bubbles = len(single_new_rows) + len(overlap_new_rows)
        if total_new_bubbles > 0:
            self.logger.info(f"第{frame_idx}帧: 为{total_new_bubbles}个新气泡分配了Y坐标")
        
        # === 评估y坐标分布并调整 ===
        if len(current_frame_df) > 5:  # 至少需要几个气泡才能评估分布
            # 计算分配后的y坐标统计特性
            y_values = current_frame_df['y'].values
            mu_y, std_y_actual = stats.norm.fit(y_values)
            
            # 比较y和x的标准差，检查是否在允许的浮动范围内
            std_ratio = std_y_actual / std_x
            self.logger.info(f"第{frame_idx}帧坐标分布: x标准差={std_x:.2f}, y标准差={std_y_actual:.2f}, 比值={std_ratio:.2f}")
            
            # 如果y方向标准差超出允许的浮动范围（0.9-1.1），则调整
            if std_ratio < 0.9 or std_ratio > 1.1:
                self.logger.info(f"第{frame_idx}帧: y方向分布需要调整（标准差比值={std_ratio:.2f}）")
                
                # 计算调整因子，使y的标准差接近x的标准差
                target_std_y = std_x
                adjust_factor = target_std_y / std_y_actual
                
                # 调整y坐标，扩展或收缩分布
                # 保持平均值不变，按比例调整与平均值的距离
                # 只调整新气泡，保持已存在气泡的连续性
                for idx, row in current_frame_df.iterrows():
                    bubble_id = row['bubble_id']
                    if bubble_id not in prev_bubbles:  # 只调整新气泡
                        y_current = current_frame_df.at[idx, 'y']
                        y_adjusted = mu_y + (y_current - mu_y) * adjust_factor
                        y_adjusted = max(0, min(y_adjusted, self.DEPTH_RANGE))
                        current_frame_df.at[idx, 'y'] = y_adjusted
                
                # 检查调整后的分布
                y_values_adjusted = current_frame_df['y'].values
                mu_y_adj, std_y_adj = stats.norm.fit(y_values_adjusted)
                self.logger.info(f"第{frame_idx}帧调整后: y均值={mu_y_adj:.2f}, 标准差={std_y_adj:.2f}, 比值={std_y_adj/std_x:.2f}")
        
        self.logger.debug(f"第{frame_idx}帧: 总共预测了{len(current_frame_df)}个气泡的3D坐标")
        return current_frame_df
    
    def save_intermediate_results(self, frame_info, frame_idx):
        """保存中间处理结果"""
        if not self.save_intermediate or not self.csv_output_dir:
            return
        
        try:
            # 确保CSV输出目录存在
            os.makedirs(self.csv_output_dir, exist_ok=True)
            
            # 保存气泡检测CSV
            csv_file = os.path.join(self.csv_output_dir, f"frame_{frame_idx:04d}.csv")
            
            if frame_info:
                # 转换为DataFrame
                df = pd.DataFrame.from_dict(frame_info, orient='index')
                
                # 确保包含3D坐标信息
                if 'bubbles_3d' in locals() and len(self.prev_frame_3d) > 0:
                    # 如果有3D信息，则更新DataFrame
                    for idx, row in df.iterrows():
                        bubble_id = int(idx)
                        # 查找对应的3D气泡信息
                        for bubble_3d in self.prev_frame_3d:
                            if bubble_3d.get('bubble_id') == bubble_id:
                                # 添加3D坐标信息
                                df.at[idx, 'x_3d'] = bubble_3d.get('x', row.get('x', 0))
                                df.at[idx, 'y_3d'] = bubble_3d.get('y', 0)  # 深度方向
                                df.at[idx, 'z_3d'] = bubble_3d.get('z', row.get('y', 0))
                                
                                # 添加尺寸信息
                                df.at[idx, 'a'] = bubble_3d.get('a', row.get('width', 10) / 2)
                                df.at[idx, 'b'] = bubble_3d.get('b', row.get('height', 10) / 2)
                                df.at[idx, 'c'] = bubble_3d.get('c', max(df.at[idx, 'a'], df.at[idx, 'b']))
                                
                                # 添加角度信息（弧度）
                                if 'angle_rad' in bubble_3d:
                                    df.at[idx, 'angle_rad'] = bubble_3d['angle_rad']
                                elif 'angle' in row:
                                    df.at[idx, 'angle_rad'] = np.deg2rad(row['angle'])
                                break
                
                # 添加帧索引
                df['frame_idx'] = frame_idx
                
                # 保存CSV
                df.to_csv(csv_file, index=True)
                self.logger.debug(f"已保存气泡信息到CSV: {csv_file}")
            
        except Exception as e:
            self.logger.warning(f"保存中间结果失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_frame_stl_with_mesh(self, bubbles_3d, frame_idx):
        """保存当前帧的STL文件，先将OBJ转换为修正后的STL，再进行合并"""
        if not bubbles_3d:
            return
        
        # 检查是否需要保存STL文件
        if not self.save_stl and not self.save_usda:
            return
        
        try:
            # 创建存放修正后STL文件的目录
            processed_stl_dir = os.path.join(self.base_path, 'processed_stl')
            os.makedirs(processed_stl_dir, exist_ok=True)
            
            # 创建单独bubble对应的文件夹
            for idx, bubble in enumerate(bubbles_3d):
                bubble_id = bubble.get('bubble_id', idx)
                bubble_dir = os.path.join(processed_stl_dir, f"bubble_{bubble_id:04d}")
                os.makedirs(bubble_dir, exist_ok=True)
            
            # 存储所有成功处理的STL文件路径
            processed_stl_files = []
            
            # 处理每个气泡
            for bubble in bubbles_3d:
                bubble_id = bubble.get('bubble_id', 0)
                
                # 为bubble添加frame_idx信息
                bubble['frame_idx'] = frame_idx
                
                # 如果有mesh_path，使用新函数将OBJ转换为STL
                if 'mesh_path' in bubble and os.path.exists(bubble['mesh_path']):
                    mesh_path = bubble['mesh_path']
                    # 确保使用绝对路径
                    if not os.path.isabs(mesh_path):
                        obj_path = os.path.abspath(mesh_path)
                    else:
                        obj_path = mesh_path
                    
                    # 检查obj文件是否存在
                    if os.path.exists(obj_path):
                        bubble_dir = os.path.join(processed_stl_dir, f"bubble_{bubble_id:04d}")
                        stl_path = self.convert_obj_to_stl(obj_path, bubble, bubble_dir)
                        if stl_path:
                            processed_stl_files.append(stl_path)
                    else:
                        self.logger.warning(f"未找到有效的OBJ文件: {obj_path}")
                else:
                    # 没有mesh_path，使用椭球体作为替代
                    self.logger.debug(f"气泡{bubble_id}没有mesh_path，使用椭球体代替")
                    
                    # 创建椭球体并保存为STL
                    try:
                        import pyvista as pv
                        import numpy as np
                        
                        # 获取气泡三维位置和尺寸
                        x_pos = bubble.get('x', 0)
                        y_pos = bubble.get('y', 0)  # 三维深度坐标
                        z_pos = bubble.get('z', 0) if 'z' in bubble else self.IMAGE_HEIGHT - bubble.get('y', 0)
                        
                        # 获取椭球体参数
                        a = bubble.get('a', bubble.get('width', 10) / 2)
                        b = bubble.get('b', bubble.get('height', 10) / 2)
                        c = bubble.get('c', max(a, b))
                        
                        # 创建椭球体
                        mesh = pv.ParametricEllipsoid(a, c, b)  # 注意PyVista的参数顺序是a,c,b
                        
                        # 如果需要，根据angle旋转
                        if 'angle_rad' in bubble and bubble['angle_rad'] != 0:
                            mesh.rotate_z(np.degrees(bubble['angle_rad']), inplace=True)
                        elif 'angle(degree)' in bubble and bubble['angle(degree)'] != 0:
                            mesh.rotate_z(bubble['angle(degree)'], inplace=True)
                        
                        # 移动到正确位置
                        mesh.translate([x_pos, y_pos, z_pos], inplace=True)
                        
                        # 保存为STL文件
                        bubble_dir = os.path.join(processed_stl_dir, f"bubble_{bubble_id:04d}")
                        os.makedirs(bubble_dir, exist_ok=True)
                        stl_filename = f"frame_{frame_idx/10000:.4f}.stl"
                        stl_path = os.path.join(bubble_dir, stl_filename)
                        mesh.save(stl_path, binary=True)
                        
                        processed_stl_files.append(stl_path)
                        self.logger.debug(f"为气泡{bubble_id}创建椭球体STL: {stl_path}")
                    except Exception as e:
                        self.logger.warning(f"为气泡{bubble_id}创建椭球体失败: {str(e)}")
            
            # 如果有成功处理的STL文件，则进行合并
            if processed_stl_files:
                try:
                    import pyvista as pv
                    
                    # 创建存放合并后STL文件的目录
                    merged_stl_dir = os.path.join(self.base_path, 'merged_stl')
                    os.makedirs(merged_stl_dir, exist_ok=True)
                    
                    # 读取所有STL文件
                    meshes = []
                    for stl_path in processed_stl_files:
                        try:
                            # 确保使用绝对路径
                            if not os.path.isabs(stl_path):
                                stl_path = os.path.abspath(stl_path)
                                
                            mesh = pv.read(stl_path)
                            if mesh.n_points > 0 and mesh.n_cells > 0:
                                meshes.append(mesh)
                        except Exception as e:
                            self.logger.warning(f"读取STL文件 {stl_path} 失败: {str(e)}")
                    
                    # 如果有有效的mesh，则进行合并
                    if meshes:
                        # 使用MultiBlock合并
                        appended = pv.MultiBlock(meshes)
                        
                        # 将MultiBlock转换为单一网格
                        try:
                            # 尝试用combine()方法合并
                            combined_mesh = appended.combine()
                            # 确保是PolyData
                            if not isinstance(combined_mesh, pv.PolyData):
                                combined_mesh = combined_mesh.extract_surface()
                        except Exception as e:
                            self.logger.warning(f"使用combine()合并失败: {str(e)}，尝试逐个合并")
                            # 如果combine()失败，尝试逐个合并
                            combined_mesh = pv.PolyData()
                            for mesh in meshes:
                                combined_mesh = combined_mesh.merge(mesh, inplace=False)
                        
                        # 最后一次清理和修复
                        combined_mesh = combined_mesh.clean(
                            point_merging=True,
                            merge_tol=1e-6,
                            lines_to_points=True,
                            polys_to_lines=True,
                            strips_to_polys=True,
                            inplace=False
                        )
                        
                        # 填补网格上的任何空洞
                        combined_mesh = combined_mesh.fill_holes(1000, inplace=False)
                        
                        # 确保所有面都是三角形
                        combined_mesh = combined_mesh.triangulate()
                        
                        # 保存合并后的STL文件
                        merged_stl_path = os.path.join(merged_stl_dir, f"frame_{frame_idx:04d}.stl")
                        combined_mesh.save(merged_stl_path, binary=True)
                        
                        # 如果需要保存USDA文件
                        if self.save_usda:
                            try:
                                from pxr import Usd, UsdGeom, Gf, Vt, Sdf
                                
                                # 创建USDA文件目录
                                usda_dir = os.path.join(self.base_path, 'usda')
                                os.makedirs(usda_dir, exist_ok=True)
                                
                                # 创建USDA文件路径
                                usda_path = os.path.join(usda_dir, f"frame_{frame_idx:04d}.usda")
                                
                                # 创建USD舞台
                                stage = Usd.Stage.CreateNew(usda_path)
                                
                                # 创建默认prim
                                default_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
                                stage.SetDefaultPrim(default_prim.GetPrim())
                                
                                # 创建网格prim
                                mesh_path = "/World/Mesh"
                                mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
                                
                                # 获取网格数据
                                points = combined_mesh.points
                                faces = combined_mesh.faces
                                
                                # 设置面片结构
                                face_vertex_counts = Vt.IntArray([3] * (len(faces) // 4))  # 每个面3个顶点
                                
                                # 构建面片顶点索引数组，确保使用Python原生int类型
                                face_vertex_indices = []
                                for i in range(0, len(faces), 4):
                                    # 将numpy.int64转换为Python int
                                    indices = [int(idx) for idx in faces[i+1:i+4]]
                                    face_vertex_indices.extend(indices)
                                face_vertex_indices = Vt.IntArray(face_vertex_indices)
                                
                                # 设置属性
                                mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
                                mesh_prim.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
                                mesh_prim.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
                                
                                # 设置插值方式
                                mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
                                mesh_prim.CreateInterpolateBoundaryAttr().Set(UsdGeom.Tokens.edgeAndCorner)
                                
                                # 保存USD文件
                                stage.Save()
                                self.logger.info(f"成功保存USDA文件: {usda_path}")
                                
                            except ImportError:
                                self.logger.warning("未找到USD库，无法保存USDA文件")
                            except Exception as e:
                                self.logger.error(f"保存USDA文件失败: {str(e)}")
                        
                        self.logger.info(f"成功合并并保存STL文件: {merged_stl_path}, 包含{len(meshes)}个气泡, "
                                        f"{combined_mesh.n_points}个顶点, {combined_mesh.n_cells}个面")
                    else:
                        self.logger.warning(f"没有有效的mesh可以合并")
                except Exception as e:
                    self.logger.error(f"合并STL文件失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                self.logger.warning(f"没有成功处理的STL文件可以合并")

        except Exception as e:
            self.logger.error(f"保存STL文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
    
    def set_mesh_level(self, level):
        """设置Pixel2Mesh网格精细度
        
        参数:
            level: 整数, 1-3之间，分别表示低、中、高三种精细度
        
        返回:
            bool: 设置是否成功
        """
        try:
            level = int(level)
            if level < 1 or level > 3:
                self.logger.warning(f"无效的网格精细度: {level}，必须在1-3之间。使用默认值2。")
                level = 2
            
            self.mesh_level = level
            self.logger.info(f"Pixel2Mesh网格精细度已设置为: level {level} (1=低, 2=中, 3=高)")
            return True
        except Exception as e:
            self.logger.error(f"设置网格精细度失败: {str(e)}")
            return False
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BFRM Configuration Module
Created by: AI4CT Team
Author: BaodI Yu (yubaodi20@ipe.ac.cn)
GitHub: https://github.com/AI4CT/BFRM
"""

import os
import json
import logging
from pathlib import Path

class Config:
    """BFRM实时处理配置类"""
    
    def __init__(self, config_file=None):
        """
        初始化配置
        
        Args:
            config_file: 可选的配置文件路径
        """
        # 基本信息
        self.project_name = "BFRM"
        self.version = "2.0.0"
        self.organization = "AI4CT"
        self.author = "BaodI Yu"
        self.email = "yubaodi20@ipe.ac.cn"
        self.github_url = "https://github.com/AI4CT/BFRM"
        
        # 模型配置
        self.yolo_model_path = self._get_default_model_path()
        
        # 输出目录配置
        self.output_dir = "results"
        self.create_output_subdirs = True
        
        # 图像尺寸配置
        self.image_width = 1280
        self.image_height = 800
        self.depth_range = 1280
        
        # 物理参数
        self.pixel_scale = 0.080128  # mm/pixel
        self.flow_volume = 5.035e-4  # 流场体积（立方米）
        
        # 处理参数
        self.max_trajectory_length = 100
        self.default_processing_speed = 5
        self.trajectory_timeout_frames = 10  # 气泡消失多少帧后移除轨迹
        
        # 默认保存选项
        self.save_intermediate_default = False
        self.save_stl_default = True
        self.save_detection_video = True
        self.save_trajectory_data = False
        
        # 3D可视化参数
        self.bubble_opacity = 0.7
        self.show_axes = True
        self.show_labels = True
        self.default_3d_view = "isometric"
        
        # YOLO检测参数
        self.yolo_conf_threshold = 0.5
        self.yolo_iou_threshold = 0.7
        self.yolo_max_det = 100
        self.yolo_imgsz = 640
        
        # 颜色配置（BGR格式）
        self.colors = {
            'single_bubble': (0, 255, 0),      # 绿色
            'overlap_bubble': (0, 165, 255),   # 橙色
            'trajectory': (255, 255, 0),       # 青色
            'text_outline': (255, 255, 255),   # 白色
            'background_overlay': (0, 0, 0),   # 黑色
        }
        
        # 字体配置
        self.font_config = {
            'font': 'FONT_HERSHEY_SIMPLEX',
            'scale': 0.7,
            'thickness': 2,
            'line_type': 'LINE_AA'
        }
        
        # 性能优化参数
        self.use_gpu = True
        self.gpu_device_id = 0
        self.num_threads = 4
        self.frame_buffer_size = 10
        
        # 日志配置
        self.log_level = "INFO"
        self.log_to_file = True
        self.log_max_files = 10
        self.log_max_size_mb = 10
        
        # 加载配置文件（如果提供）
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # 初始化创建必要目录
        self._create_directories()
        
        # 验证配置
        self._validate_config()
    
    def _get_default_model_path(self):
        """获取默认YOLO模型路径"""
        # 尝试多个可能的模型位置
        possible_paths = [
            "model/yolo11x-obb.pt",
            "models/yolo11x-obb.pt", 
            "C:/codebase/yolo/model/yolo11x-obb.pt",
            os.path.join(os.path.dirname(__file__), "model", "yolo11x-obb.pt"),
            os.path.join(os.path.dirname(__file__), "models", "yolo11x-obb.pt")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都不存在，返回默认路径
        return "model/yolo11x-obb.pt"
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.output_dir,
            "logs",
            "temp"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_config(self):
        """验证配置参数"""
        logger = logging.getLogger(__name__)
        
        # 验证图像尺寸
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("图像尺寸必须大于0")
        
        # 验证像素比例
        if self.pixel_scale <= 0:
            raise ValueError("像素比例必须大于0")
        
        # 验证处理速度
        if not (1 <= self.default_processing_speed <= 10):
            logger.warning(f"默认处理速度 {self.default_processing_speed} 超出范围，重置为5")
            self.default_processing_speed = 5
        
        # 验证透明度
        if not (0 <= self.bubble_opacity <= 1):
            logger.warning(f"气泡透明度 {self.bubble_opacity} 超出范围，重置为0.7")
            self.bubble_opacity = 0.7
        
        # 验证YOLO参数
        if not (0 <= self.yolo_conf_threshold <= 1):
            logger.warning(f"YOLO置信度阈值 {self.yolo_conf_threshold} 超出范围，重置为0.5")
            self.yolo_conf_threshold = 0.5
        
        if not (0 <= self.yolo_iou_threshold <= 1):
            logger.warning(f"YOLO IoU阈值 {self.yolo_iou_threshold} 超出范围，重置为0.7")
            self.yolo_iou_threshold = 0.7
        
        logger.info("配置验证完成")
    
    def load_from_file(self, config_file):
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logging.info(f"成功加载配置文件: {config_file}")
            
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            raise e
    
    def save_to_file(self, config_file):
        """保存配置到文件"""
        try:
            # 获取所有公共属性
            config_data = {}
            for key, value in self.__dict__.items():
                if not key.startswith('_'):
                    config_data[key] = value
            
            # 确保目录存在
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            # 保存JSON文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            
            logging.info(f"配置已保存到: {config_file}")
            
        except Exception as e:
            logging.error(f"保存配置文件失败: {str(e)}")
            raise e
    
    def update_yolo_model_path(self, path):
        """更新YOLO模型路径"""
        if os.path.exists(path):
            self.yolo_model_path = path
            logging.info(f"更新YOLO模型路径: {path}")
        else:
            raise FileNotFoundError(f"YOLO模型文件不存在: {path}")
    
    def update_output_dir(self, path):
        """更新输出目录"""
        self.output_dir = path
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"更新输出目录: {path}")
    
    def get_log_config(self):
        """获取日志配置"""
        return {
            'level': getattr(logging, self.log_level.upper()),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'to_file': self.log_to_file,
            'max_files': self.log_max_files,
            'max_size_mb': self.log_max_size_mb
        }
    
    def get_yolo_config(self):
        """获取YOLO配置"""
        return {
            'model_path': self.yolo_model_path,
            'conf': self.yolo_conf_threshold,
            'iou': self.yolo_iou_threshold,
            'max_det': self.yolo_max_det,
            'imgsz': self.yolo_imgsz,
            'device': 'cuda:0' if self.use_gpu else 'cpu',
            'tracker': 'botsort.yaml'
        }
    
    def get_visualization_config(self):
        """获取可视化配置"""
        return {
            'bubble_opacity': self.bubble_opacity,
            'show_axes': self.show_axes,
            'show_labels': self.show_labels,
            'default_view': self.default_3d_view,
            'colors': self.colors.copy()
        }
    
    def get_processing_config(self):
        """获取处理配置"""
        return {
            'max_trajectory_length': self.max_trajectory_length,
            'trajectory_timeout_frames': self.trajectory_timeout_frames,
            'save_intermediate': self.save_intermediate_default,
            'save_stl': self.save_stl_default,
            'processing_speed': self.default_processing_speed
        }
    
    def get_physical_config(self):
        """获取物理参数配置"""
        return {
            'pixel_scale': self.pixel_scale,
            'flow_volume': self.flow_volume,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'depth_range': self.depth_range
        }
    
    def __str__(self):
        """返回配置的字符串表示"""
        return f"""
BFRM Configuration v{self.version}
================================
Project: {self.project_name}
Author: {self.author} ({self.email})
GitHub: {self.github_url}

Model Configuration:
- YOLO Model: {self.yolo_model_path}
- Use GPU: {self.use_gpu}
- Device ID: {self.gpu_device_id}

Image Configuration:
- Resolution: {self.image_width}×{self.image_height}
- Depth Range: {self.depth_range}
- Pixel Scale: {self.pixel_scale} mm/pixel

Processing Configuration:
- Default Speed: {self.default_processing_speed}
- Max Trajectory Length: {self.max_trajectory_length}
- Save Intermediate: {self.save_intermediate_default}
- Save STL: {self.save_stl_default}

Output Configuration:
- Output Directory: {self.output_dir}
"""

# 创建默认配置实例
default_config = Config()

# 配置文件路径
CONFIG_FILE_PATH = "config/bfrm_config.json"

def load_config(config_file=None):
    """加载配置"""
    if config_file is None:
        config_file = CONFIG_FILE_PATH
    
    if os.path.exists(config_file):
        return Config(config_file)
    else:
        # 创建并保存默认配置
        config = Config()
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        config.save_to_file(config_file)
        return config

def save_config(config, config_file=None):
    """保存配置"""
    if config_file is None:
        config_file = CONFIG_FILE_PATH
    
    config.save_to_file(config_file)

# dataset root
DATASET_ROOT = "C:/Users/Administrator/Desktop/BFRM/datasets"
# SHAPENET_ROOT = os.path.join(DATASET_ROOT, "shapenet")
# SHAPENET_ROOT = os.path.join('/root/autodl-tmp/3Dbubbledataset')
SHAPENET_ROOT = os.path.join('C:/DataSet_DOOR/dataset_Reconstruction_normalization')
IMAGENET_ROOT = os.path.join(DATASET_ROOT, "imagenet")

# ellipsoid path
ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "ellipsoid/info_ellipsoid.dat")
# pretrained weights path
PRETRAINED_WEIGHTS_PATH = {
    "vgg16": os.path.join(DATASET_ROOT, "pretrained/vgg16-397923af.pth"),
    "resnet50": os.path.join(DATASET_ROOT, "pretrained/resnet50-19c8e357.pth"),
    # "resnet50": os.path.join(DATASET_ROOT, "pretrained/resnet.pth.tar"),
    "vgg16p2m": os.path.join(DATASET_ROOT, "pretrained/vgg16-p2m.pth"),
}

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 128
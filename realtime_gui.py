#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BFRM Realtime GUI Module
Created by: AI4CT Team
Author: BaodI Yu (yubaodi20@ipe.ac.cn)
"""

import os
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                            QPushButton, QLabel, QProgressBar, QStatusBar,
                            QGroupBox, QSplitter, QFileDialog, QMenuBar, 
                            QMenu, QSlider, QCheckBox, QSpinBox, QComboBox,
                            QMessageBox, QApplication, QFrame, QTextEdit,
                            QTabWidget, QGridLayout, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon, QAction, QFont
import pyvista as pl
from pyvistaqt import BackgroundPlotter
import logging
from realtime_processor import RealtimeProcessor
from config import Config

class VideoDisplayWidget(QLabel):
    """自定义视频显示控件"""
    
    def __init__(self, parent=None, default_message="等待视频输入..."):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #333333;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        self.setText(default_message)
        self.video_width = 640
        self.video_height = 480
        self.current_frame = None  # 新增：缓存当前帧
        self.default_message = default_message
        
    def set_video_resolution(self, width, height):
        """设置视频分辨率"""
        self.video_width = width
        self.video_height = height
        
    def display_frame(self, frame):
        """显示视频帧"""
        if frame is None:
            self.current_frame = None
            self.setText(self.default_message)
            return
        try:
            self.current_frame = frame.copy()  # 缓存当前帧
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.setPixmap(pixmap.scaled(
                self.width(), 
                self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        except Exception as e:
            self.current_frame = None
            self.clear()
            self.setText(f"显示错误: {str(e)}")
    
    def resizeEvent(self, event):
        """窗口尺寸变化时自动刷新显示"""
        if self.current_frame is not None:
            self.display_frame(self.current_frame)
        else:
            self.setText(self.default_message)
        super().resizeEvent(event)

class RealtimeMainWindow(QMainWindow):
    """BFRM实时处理主窗口"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processor = None
        self.is_processing = False
        self.current_frame = None
        self.frame_count = 0
        self.start_time = None
        
        # 初始化UI
        self.init_ui()
        
        # 初始化实时处理器
        self.processor = RealtimeProcessor(config, self)
        
        # 连接信号
        self.processor.frame_processed.connect(self.update_display)
        self.processor.error_occurred.connect(self.handle_error)
        self.processor.processing_finished.connect(self.on_processing_finished)
        self.processor.status_updated.connect(self.status_label.setText)
        
        # 创建状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)  # 每100ms更新一次状态
        
        # 1. 在RealtimeMainWindow.__init__或setup_3d_scene中初始化
        self.bubble_actors = []
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("BFRM - Realtime Bubble Flow Reconstruction v2.0.0")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1000, 600)
        icon_path = os.path.join("icons", "bubble.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.create_menu_bar()
        self.create_toolbar()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        # 只保留左侧面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)
        # 先添加右上角按钮控件，确保按钮已初始化
        self.add_top_right_controls(main_layout)
        self.create_status_bar()
        self.update_button_states(False)
        # 初始化所有显示数据
        self.init_display_values()
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        # 打开视频
        open_video_action = QAction(QIcon("icons/open.png"), '打开视频(&O)', self)
        open_video_action.setShortcut('Ctrl+O')
        open_video_action.setStatusTip('打开视频文件进行处理')
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)
        
        file_menu.addSeparator()
        
        # 导出STL
        export_stl_action = QAction(QIcon("icons/export.png"), '导出STL(&S)', self)
        export_stl_action.setShortcut('Ctrl+S')
        export_stl_action.setStatusTip('导出当前帧的STL文件')
        export_stl_action.triggered.connect(self.export_stl)
        self.export_stl_action = export_stl_action
        file_menu.addAction(export_stl_action)
        
        # 导出所有STL文件
        export_all_stl_action = QAction('导出所有STL文件(&A)', self)
        export_all_stl_action.setStatusTip('导出所有已处理帧的STL文件')
        export_all_stl_action.triggered.connect(self.export_all_stl)
        self.export_all_stl_action = export_all_stl_action
        file_menu.addAction(export_all_stl_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction('退出(&X)', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('退出程序')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 处理菜单
        process_menu = menubar.addMenu('处理(&P)')
        
        # 开始/停止处理
        self.start_stop_action = QAction(QIcon("icons/play.png"), '开始处理(&S)', self)
        self.start_stop_action.setShortcut('Space')
        self.start_stop_action.setStatusTip('开始或停止实时处理')
        self.start_stop_action.triggered.connect(self.toggle_processing)
        process_menu.addAction(self.start_stop_action)
        
        # 暂停处理
        self.pause_action = QAction(QIcon("icons/pause.png"), '暂停处理(&P)', self)
        self.pause_action.setShortcut('Ctrl+P')
        self.pause_action.setStatusTip('暂停处理')
        self.pause_action.triggered.connect(self.pause_processing)
        process_menu.addAction(self.pause_action)
        
        # 重置
        reset_action = QAction(QIcon("icons/reset.png"), '重置(&R)', self)
        reset_action.setShortcut('Ctrl+R')
        reset_action.setStatusTip('重置所有状态')
        reset_action.triggered.connect(self.reset_processing)
        process_menu.addAction(reset_action)
        
        process_menu.addSeparator()
        
        # 处理设置
        process_settings_action = QAction('处理设置(&T)', self)
        process_settings_action.setStatusTip('配置处理参数')
        process_settings_action.triggered.connect(self.show_process_settings)
        process_menu.addAction(process_settings_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图(&V)')
        
        # 3D视角选择
        view_submenu = view_menu.addMenu('3D视角(&3)')
        
        self.view_actions = {}
        views = [
            ('透视图(&I)', 'isometric'),
            ('正视图(&F)', 'front'),
            ('侧视图(&S)', 'side'),
            ('俯视图(&T)', 'top')
        ]
        
        for name, view_type in views:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, vt=view_type: self.change_3d_view(vt))
            view_submenu.addAction(action)
            self.view_actions[view_type] = action
        
        view_menu.addSeparator()
        
        # 显示选项
        self.show_axes_action = QAction('显示坐标轴(&A)', self, checkable=True)
        self.show_axes_action.setChecked(True)
        self.show_axes_action.triggered.connect(self.toggle_axes)
        view_menu.addAction(self.show_axes_action)
        
        self.show_labels_action = QAction('显示标签(&L)', self, checkable=True)
        self.show_labels_action.setChecked(True)
        self.show_labels_action.triggered.connect(self.toggle_labels)
        view_menu.addAction(self.show_labels_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        # 关于
        about_action = QAction('关于BFRM(&A)', self)
        about_action.setStatusTip('关于BFRM系统')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # GitHub链接
        github_action = QAction('GitHub项目主页(&G)', self)
        github_action.setStatusTip('访问GitHub项目主页')
        github_action.triggered.connect(self.open_github)
        help_menu.addAction(github_action)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = self.addToolBar('主工具栏')
        toolbar.setIconSize(QSize(32, 32))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        
        # 打开视频
        toolbar.addAction(QIcon("icons/open.png"), '打开视频', self.open_video)
        toolbar.addSeparator()
        
        # 处理控制（右对齐）
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # 右侧控制按钮
        self.play_action_btn = QPushButton("开始")
        self.play_action_btn.setIcon(QIcon("icons/play.png"))
        self.play_action_btn.clicked.connect(self.toggle_processing)
        toolbar.addWidget(self.play_action_btn)
        self.pause_action_btn = QPushButton("暂停")
        self.pause_action_btn.setIcon(QIcon("icons/pause.png"))
        self.pause_action_btn.clicked.connect(self.pause_processing)
        toolbar.addWidget(self.pause_action_btn)
        self.stop_action_btn = QPushButton("停止")
        self.stop_action_btn.setIcon(QIcon("icons/stop.png"))
        self.stop_action_btn.clicked.connect(self.stop_processing)
        toolbar.addWidget(self.stop_action_btn)
        
        toolbar.addSeparator()
        # 导出
        toolbar.addAction(QIcon("icons/export.png"), '导出STL', self.export_stl)
        
        # 设置初始状态
        self.pause_action_tb = self.pause_action_btn
        self.stop_action = self.stop_action_btn
        self.play_action = self.play_action_btn
        self.pause_action_tb.setEnabled(False)
        self.stop_action.setEnabled(False)
    
    def create_left_panel(self):
        """创建左侧面板"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        
        # 视频文件选择
        file_group = QGroupBox("视频文件")
        file_layout = QVBoxLayout(file_group)
        
        file_select_layout = QHBoxLayout()
        self.file_label = QLabel("请选择视频文件...")
        self.file_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.open_video)
        
        file_select_layout.addWidget(self.file_label, 1)
        file_select_layout.addWidget(self.browse_btn)
        file_layout.addLayout(file_select_layout)
        
        # 视频信息显示
        self.video_info_label = QLabel("视频信息: 未加载任何视频")
        self.video_info_label.setStyleSheet("font-size: 10px; color: #888;")
        file_layout.addWidget(self.video_info_label)
        
        control_layout.addWidget(file_group)
        
        left_layout.addWidget(control_group)
        
        # 实时视频流区域
        video_group = QGroupBox("实时视频流")
        video_layout = QVBoxLayout(video_group)

        # 横向并排两个视频显示控件
        video_hbox = QHBoxLayout()
        # 原始视频
        self.original_video_widget = VideoDisplayWidget(default_message="原始视频 - 等待视频输入...")
        self.original_video_widget.setMinimumSize(320, 240)
        video_hbox.addWidget(self.original_video_widget)
        # 处理后视频
        self.video_widget = VideoDisplayWidget(default_message="处理后视频 - 等待处理...")
        self.video_widget.setMinimumSize(320, 240)
        video_hbox.addWidget(self.video_widget)
        video_layout.addLayout(video_hbox)

        # 帧数、时间、进度条
        info_hbox = QHBoxLayout()
        self.frame_info_label = QLabel("帧: 0/0")
        self.time_info_label = QLabel("时间: 00:00")
        info_hbox.addWidget(self.frame_info_label)
        info_hbox.addStretch()
        info_hbox.addWidget(self.time_info_label)
        video_layout.addLayout(info_hbox)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("处理进度: %p%")
        self.progress_bar.setValue(0)
        video_layout.addWidget(self.progress_bar)

        left_layout.addWidget(video_group)
        left_layout.setStretch(0, 0)  # 控制面板不拉伸
        left_layout.setStretch(1, 1)  # 视频显示区域拉伸
        return left_panel
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        
        # 主状态信息
        self.status_label = QLabel("就绪 - 请选择视频文件")
        self.status_bar.addWidget(self.status_label)
        
        # 添加分隔符
        self.status_bar.addWidget(QLabel("|"))
        
        # 处理进度
        self.process_status_label = QLabel("未开始")
        self.status_bar.addWidget(self.process_status_label)
        
        # FPS显示（两位小数）
        self.fps_status_label = QLabel("FPS: 0.00")
        self.status_bar.addPermanentWidget(self.fps_status_label)
        
        # 处理时间显示
        self.processing_time_label = QLabel("处理时间: 0.0ms")
        self.status_bar.addPermanentWidget(self.processing_time_label)
        
        # 帧索引显示
        self.frame_idx_label = QLabel("帧: 0")
        self.status_bar.addPermanentWidget(self.frame_idx_label)
        
        # 时间显示（总处理时长）
        self.time_status_label = QLabel("总处理时长: 0.0s")
        self.status_bar.addPermanentWidget(self.time_status_label)
    
    def update_button_states(self, processing):
        """更新按钮状态"""
        # 工具栏按钮
        if processing:
            self.play_action.setIcon(QIcon("icons/stop.png"))
            self.play_action.setText("停止")
            self.pause_action_tb.setEnabled(True)
            self.stop_action.setEnabled(True)
            # 右上角按钮
            self.play_action_btn.setText("停止")
            self.play_action_btn.setIcon(QIcon("icons/stop.png"))
            self.pause_action_btn.setEnabled(True)
            self.stop_action_btn.setEnabled(True)
            # 菜单动作
            self.start_stop_action.setText("停止处理(&S)")
            self.start_stop_action.setIcon(QIcon("icons/stop.png"))
            self.pause_action.setEnabled(True)
        else:
            self.play_action.setIcon(QIcon("icons/play.png"))
            self.play_action.setText("开始")
            self.pause_action_tb.setEnabled(False)
            self.stop_action.setEnabled(False)
            # 右上角按钮
            self.play_action_btn.setText("开始")
            self.play_action_btn.setIcon(QIcon("icons/play.png"))
            self.pause_action_btn.setEnabled(False)
            self.stop_action_btn.setEnabled(False)
            # 菜单动作
            self.start_stop_action.setText("开始处理(&S)")
            self.start_stop_action.setIcon(QIcon("icons/play.png"))
            self.pause_action.setEnabled(False)
        # 文件选择控制
        self.browse_btn.setEnabled(not processing)
    
    def toggle_axes(self, checked=None):
        """切换坐标轴显示 - 保留但不做任何操作"""
        pass
    
    def toggle_labels(self, checked=None):
        """切换标签显示 - 保留但不做任何操作"""
        pass
    
    def init_display_values(self):
        """初始化所有显示数据为默认值"""
        # 设置初始状态文本
        self.process_status_label.setText("未开始")
        self.fps_status_label.setText("0.00")
        self.frame_info_label.setText("帧: 0/0")
        self.time_info_label.setText("时间: 00:00")
        self.progress_bar.setValue(0)
        
        # 设置状态栏初始文本
        self.status_label.setText("就绪 - 请选择视频文件")
        
        # 清除视频显示
        self.original_video_widget.setText("原始视频 - 等待视频输入...")
        self.video_widget.setText("处理后视频 - 等待处理...")
    
    def open_video(self):
        """打开视频文件"""
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if not file_path:
            return
        
        try:
            # 初始化界面状态
            self.init_display_values()
            
            # 设置文件路径标签
            self.file_label.setText(os.path.basename(file_path))
            self.file_label.setToolTip(file_path)
            
            # 创建处理器（如果不存在）
            if not self.processor:
                self.processor = RealtimeProcessor(self.config)
                
                # 连接信号
                self.processor.frame_processed.connect(self.update_display)
                self.processor.error_occurred.connect(self.handle_error)
                self.processor.processing_finished.connect(self.on_processing_finished)
                self.processor.status_updated.connect(self.status_label.setText)
            
            # 加载视频
            self.processor.load_video(file_path)
            
            # 更新视频信息
            video_info = f"视频: {os.path.basename(file_path)} | "
            video_info += f"分辨率: {int(self.processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(self.processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | "
            video_info += f"帧数: {self.processor.total_frames} | "
            video_info += f"帧率: {self.processor.video_fps:.2f} FPS"
            self.video_info_label.setText(video_info)
            
            # 设置视频显示控件的分辨率
            self.video_widget.set_video_resolution(
                int(self.processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            self.original_video_widget.set_video_resolution(
                int(self.processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            
            # 更新按钮状态
            self.update_button_states(True)
            
            # 更新状态栏
            self.status_label.setText(f"视频已加载: {os.path.basename(file_path)}")
            
            # 更新进度条
            self.progress_bar.setValue(0)
            
            # 更新视频显示为等待处理状态
            self.original_video_widget.setText("原始视频 - 就绪，等待处理...")
            self.video_widget.setText("处理后视频 - 就绪，等待处理...")

            # 自动开始处理
            self.start_processing()
            
        except Exception as e:
            self.logger.error(f"打开视频失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"打开视频失败:\n{str(e)}")
            
    def toggle_processing(self):
        """切换处理状态"""
        if not self.is_processing:
            self.start_processing()
        else:
            self.stop_processing()
    
    def start_processing(self):
        """开始处理"""
        if not self.processor or not self.processor.video_path:
            QMessageBox.warning(self, "警告", "请先选择视频文件！")
            return
        
        try:
            # 记录开始时间
            self.start_time = time.time()
            self.frame_count = 0
            
            # 开始处理
            self.processor.start_processing()
            self.is_processing = True
            
            # 更新UI状态
            self.update_button_states(True)
            self.status_label.setText("正在处理...")
            self.process_status_label.setText("处理中")
            
            self.logger.info("开始实时处理")
            
        except Exception as e:
            self.logger.error(f"启动处理失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动处理失败:\n{str(e)}")
            self.status_label.setText("启动失败")
    
    def pause_processing(self):
        """暂停处理"""
        if self.processor and self.is_processing:
            self.processor.pause_processing()
            self.status_label.setText("已暂停")
            self.process_status_label.setText("已暂停")
    
    def stop_processing(self):
        """停止处理"""
        if self.processor and self.is_processing:
            self.processor.stop_processing()
            self.is_processing = False
            
            # 更新UI状态
            self.update_button_states(False)
            self.status_label.setText("已停止")
            self.process_status_label.setText("已停止")
            
            self.logger.info("停止实时处理")
    
    def reset_processing(self):
        """重置处理"""
        # 先停止处理
        if self.is_processing:
            self.stop_processing()
        
        # 重置处理器
        if self.processor:
            self.processor.reset()
        
        # 重置UI显示
        self.init_display_values()
        
        # 重置计数器
        self.frame_count = 0
        self.start_time = None
        
        self.status_label.setText("已重置")
        
    def update_display(self, frame_data):
        """更新显示"""
        try:
            # 更新FPS显示（两位小数）
            fps = frame_data.get('fps', 0.0)
            self.fps_status_label.setText(f"FPS: {fps:.2f}")
            
            # 更新进度条
            progress = frame_data.get('progress', 0)
            self.progress_bar.setValue(int(progress))
            
            # 更新处理时间
            processing_time = frame_data.get('processing_time', 0)
            self.processing_time_label.setText(f"处理时间: {processing_time*1000:.1f}ms")
            
            # 更新帧索引
            frame_idx = frame_data.get('frame_idx', 0)
            self.frame_count = frame_idx + 1  # 新增：实时更新已处理帧数
            total_frames = self.processor.total_frames if self.processor else 0
            self.frame_idx_label.setText(f"帧: {frame_idx}")
            self.frame_info_label.setText(f"帧: {frame_idx}/{total_frames}")
            
            # 更新时间（进度条上方和右侧）
            if hasattr(self.processor, 'video_fps') and self.processor.video_fps > 0:
                seconds = frame_idx / self.processor.video_fps
                total_seconds = total_frames / self.processor.video_fps if total_frames else 0
                minutes = int(seconds / 60)
                seconds_int = int(seconds % 60)
                total_minutes = int(total_seconds / 60)
                total_seconds_int = int(total_seconds % 60)
                time_str = f"{minutes:02d}:{seconds_int:02d}/{total_minutes:02d}:{total_seconds_int:02d}"
            else:
                time_str = "00:00/00:00"
            self.time_info_label.setText(f"时间: {time_str}")

            # 状态栏最右侧显示总处理时长（单位：秒，保留一位小数）
            if self.start_time:
                elapsed = time.time() - self.start_time
                self.time_status_label.setText(f"总处理时长: {elapsed:.1f}s")
            else:
                self.time_status_label.setText("总处理时长: 0.0s")
            
            # 显示原始视频帧
            if 'original_frame' in frame_data:
                rgb_frame = cv2.cvtColor(frame_data['original_frame'], cv2.COLOR_BGR2RGB)
                self.original_video_widget.display_frame(rgb_frame)
            else:
                self.original_video_widget.setText("未获取到原始视频帧")
            
            # 显示处理后视频帧
            if 'processed_frame' in frame_data:
                rgb_processed = cv2.cvtColor(frame_data['processed_frame'], cv2.COLOR_BGR2RGB)
                self.video_widget.display_frame(rgb_processed)
            else:
                self.video_widget.setText("未获取到处理后视频帧")
        except Exception as e:
            self.logger.error(f"更新显示时出错: {str(e)}")
    
    def update_time_info(self):
        """更新时间信息"""
        if not self.processor or not hasattr(self.processor, 'current_frame_idx'):
            self.time_info_label.setText("时间: 00:00")
            self.frame_info_label.setText("帧: 0/0")
            return
            
        try:
            frame_idx = self.processor.current_frame_idx
            total_frames = self.processor.total_frames
            
            # 计算时间
            if hasattr(self.processor, 'video_fps') and self.processor.video_fps > 0:
                seconds = frame_idx / self.processor.video_fps
                minutes = int(seconds / 60)
                seconds = int(seconds % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
            else:
                time_str = "00:00"
            
            # 更新标签
            self.time_info_label.setText(f"时间: {time_str}")
            self.frame_info_label.setText(f"帧: {frame_idx}/{total_frames}")
            
        except Exception as e:
            self.logger.error(f"更新时间信息失败: {str(e)}")
    
    def update_3d_display(self, bubbles_3d):
        """更新3D显示"""
        if not hasattr(self, 'plotter'):
            return
        try:
            # 1. 清除之前的气泡actor
            if not hasattr(self, 'bubble_actors'):
                self.bubble_actors = []
            for actor in self.bubble_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.bubble_actors.clear()
            # 2. 清除之前的label（只在show_labels为True时管理）
            if not hasattr(self, 'bubble_labels'):
                self.bubble_labels = []
            for label in self.bubble_labels:
                try:
                    self.plotter.remove_actor(label)
                except Exception:
                    pass
            self.bubble_labels.clear()
            # 3. 添加新气泡
            for bubble in bubbles_3d:
                mesh = self.create_bubble_mesh(bubble)
                color = '#4CAF50' if bubble.get('type') == 'single' else '#FF9800' if bubble.get('type') == 'overlap' else '#2196F3'
                actor = self.plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=self.config.bubble_opacity,
                    show_edges=False
                )
                self.bubble_actors.append(actor)
            # 4. 添加label（只在show_labels为True时）
            if self.config.show_labels:
                for bubble in bubbles_3d:
                    label = self.plotter.add_point_labels(
                        points=[[bubble.get('x',0), bubble.get('y',0), bubble.get('z',0)]],
                        labels=[str(bubble.get('bubble_id',''))],
                        point_size=1,
                        font_size=10,
                        text_color='black',
                        shape_opacity=0.8
                    )
                    self.bubble_labels.append(label)
            # 5. 强制处理Qt事件，防止假死
            QApplication.processEvents()
        except Exception as e:
            self.logger.error(f"更新3D显示失败: {str(e)}")
    
    def create_bubble_mesh(self, bubble):
        """优先用Pixel2Mesh mesh_path加载，否则用椭球体"""
        if 'mesh_path' in bubble and bubble['mesh_path'] and os.path.exists(bubble['mesh_path']):
            try:
                mesh = pl.read(bubble['mesh_path'])
                # 只对x方向做scale，y/z方向保持1.0，防止mesh变形
                x_len = mesh.bounds[1] - mesh.bounds[0]
                target_x_len = bubble.get('target_x_len', bubble.get('width', 1))
                scale_factor = target_x_len / x_len if x_len > 0 else 1.0
                mesh.translate(-np.array(mesh.center), inplace=True)
                mesh.scale([scale_factor, scale_factor, scale_factor], inplace=True)
                mesh.translate([bubble.get('x',0), bubble.get('y',0), bubble.get('z',0)], inplace=True)
                return mesh
            except Exception as e:
                msg = f"加载Pixel2Mesh mesh失败，回退为椭球体: {str(e)}"
                self.logger.warning(msg)
        # 回退为椭球体
        center = [bubble.get('x',0), bubble.get('y',0), bubble.get('z',0)]
        a, b, c = bubble.get('a',1), bubble.get('b',1), bubble.get('c',1)
        ellipsoid = pl.ParametricEllipsoid(a, b, c)
        if 'angle_rad' in bubble and bubble['angle_rad'] != 0:
            ellipsoid.rotate_z(np.degrees(bubble['angle_rad']))
        ellipsoid.translate(center)
        return ellipsoid
    
    def change_3d_view(self, view_type):
        """改变3D视角"""
        if not hasattr(self, 'plotter'):
            return
        
        try:
            if view_type == "front":
                self.plotter.camera_position = 'yz'
            elif view_type == "side":
                self.plotter.camera_position = 'xz'
            elif view_type == "top":
                self.plotter.camera_position = 'xy'
            else:  # isometric
                self.plotter.camera_position = 'isometric'
            
            self.plotter.reset_camera()
            
        except Exception as e:
            self.logger.error(f"更改3D视角失败: {str(e)}")
    
    def export_stl(self):
        """导出当前帧的STL文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存STL文件",
            f"bubble_flow_frame_{self.frame_count:04d}.stl",
            "STL文件 (*.stl);;所有文件 (*)"
        )
        
        if file_path:
            try:
                if self.processor.export_current_stl(file_path):
                    self.status_label.setText(f"STL已导出: {os.path.basename(file_path)}")
                    QMessageBox.information(self, "成功", f"STL文件已成功导出到:\n{file_path}")
                else:
                    QMessageBox.warning(self, "失败", "STL文件导出失败")
            except Exception as e:
                self.logger.error(f"导出STL失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"导出STL文件时出错:\n{str(e)}")
    
    def export_all_stl(self):
        """导出所有已处理帧的STL文件"""
        if not self.processor or not self.processor.output_dir:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return
        
        stl_dir = os.path.join(self.processor.output_dir, "stl_frames")
        if not os.path.exists(stl_dir) or not os.listdir(stl_dir):
            QMessageBox.information(self, "提示", "没有找到已生成的STL文件")
            return
        
        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            ""
        )
        
        if export_dir:
            try:
                import shutil
                
                # 创建目标目录
                target_dir = os.path.join(export_dir, f"BFRM_STL_Export_{time.strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(target_dir, exist_ok=True)
                
                # 复制所有STL文件
                stl_files = [f for f in os.listdir(stl_dir) if f.endswith('.stl')]
                
                for stl_file in stl_files:
                    src = os.path.join(stl_dir, stl_file)
                    dst = os.path.join(target_dir, stl_file)
                    shutil.copy2(src, dst)
                
                QMessageBox.information(
                    self, 
                    "成功", 
                    f"成功导出 {len(stl_files)} 个STL文件到:\n{target_dir}"
                )
                
                # 询问是否打开目标目录
                reply = QMessageBox.question(
                    self,
                    "打开目录",
                    "是否要打开导出目录？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    os.startfile(target_dir)  # Windows
                    
            except Exception as e:
                self.logger.error(f"批量导出STL失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"批量导出STL文件时出错:\n{str(e)}")
    
    def show_process_settings(self):
        """显示处理设置对话框"""
        # TODO: 实现详细的处理设置对话框
        QMessageBox.information(self, "提示", "处理设置功能正在开发中...")
    
    def update_status(self):
        """定期更新状态信息"""
        if self.processor and self.is_processing:
            # 更新各种状态显示
            pass
    
    def show_about(self):
        """显示关于对话框"""
        about_text = f"""
        <h2>BFRM - Bubble Flow Reconstruction and Measurement</h2>
        <h3>Realtime Version 2.0.0</h3>
        
        <p><b>气泡流重建与测量系统</b></p>
        
        <p>该系统用于实时处理气泡流图像，执行气泡检测和追踪，
        提供气泡流分析解决方案。</p>
        
        <p><b>开发团队:</b> AI4CT (Artificial Intelligence for Computational tomography)</p>
        <p><b>作者:</b> BaodI Yu (yubaodi20@ipe.ac.cn)</p>
        <p><b>GitHub:</b> <a href="https://github.com/AI4CT/BFRM">https://github.com/AI4CT/BFRM</a></p>
        
        <p><b>主要功能:</b></p>
        <ul>
        <li>实时气泡检测与追踪</li>
        <li>STL文件导出</li>
        <li>实时统计分析</li>
        <li>双视频显示</li>
        </ul>
        
        <p><b>技术栈:</b></p>
        <ul>
        <li>PyQt6 - 用户界面</li>
        <li>YOLO - 气泡检测</li>
        <li>OpenCV - 图像处理</li>
        <li>NumPy, Pandas - 数据处理</li>
        </ul>
        
        <p>© 2024 AI4CT Team. All Rights Reserved.</p>
        """
        
        QMessageBox.about(self, "关于 BFRM", about_text)
    
    def open_github(self):
        """打开GitHub项目页面"""
        import webbrowser
        webbrowser.open("https://github.com/AI4CT/BFRM")
    
    def handle_error(self, error_msg):
        """处理错误"""
        self.logger.error(f"处理错误: {error_msg}")
        self.status_label.setText(f"错误: {error_msg}")
        
        # 显示错误对话框
        QMessageBox.critical(self, "处理错误", f"处理过程中发生错误:\n{error_msg}")
    
    def on_processing_finished(self):
        """处理完成回调"""
        self.is_processing = False
        self.update_button_states(False)
        self.status_label.setText("处理完成")
        self.process_status_label.setText("已完成")
        
        # 显示完成对话框
        QMessageBox.information(
            self, 
            "处理完成", 
            f"视频处理已完成！\n\n"
            f"总共处理帧数: {self.frame_count}\n"
            f"输出目录: {self.processor.output_dir}"
        )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 如果正在处理，询问是否确定退出
        if self.is_processing:
            reply = QMessageBox.question(
                self,
                "确认退出",
                "正在处理中，确定要退出吗？\n处理将被中断。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        # 停止处理
        if self.processor:
            self.processor.stop_processing()
            self.processor.cleanup()
        
        # 添加退出日志
        self.logger.info("BFRM Realtime 主窗口已关闭")
        
        # 接受关闭事件
        event.accept()

    def add_top_right_controls(self, main_layout):
        """在界面右上角添加开始、暂停、停止按钮"""
        # 删除右上角按钮，不再添加任何控件
        pass

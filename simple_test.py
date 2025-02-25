import sys
import os
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QMenu, QToolBar, QProgressBar,
                             QFrame, QSizePolicy, QFileDialog, QPushButton, QCheckBox,
                             QSlider, QMessageBox, QGroupBox, QStatusBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QAction, QActionGroup, QImage, QColor

class SimpleTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("气泡流检测与重构 - 简化测试")
        self.resize(1200, 800)
        
        # 初始化变量
        self.content_labels = {
            'flow': None,
            'detection': None,
            '3d': None,
            'empty': None
        }
        self.log_history = []
        self.is_playing = False
        self.playback_speed = 1.0
        self.current_frame_index = 0
        self.frames = []  # 存储测试帧
        
        # 初始化UI
        self.init_ui()
        
        # 添加程序启动日志
        self.add_log("程序启动完成")
        self.add_log("这是一个简化版测试程序，用于测试布局功能")
        
    def init_ui(self):
        # 创建中心部件和主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建工具栏
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # 添加视频控制按钮
        self.add_video_controls(toolbar)
        
        # 创建主显示区域
        self.create_main_display_area()
        
        # 创建状态栏
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("就绪")
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(15)
        self.statusbar.addPermanentWidget(self.progress_bar, 1)
        
        # 显示窗口
        self.show()
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        # 创建"加载文件夹"动作
        load_folder_action = QAction('加载文件夹', self)
        load_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(load_folder_action)
        
        # 创建"加载视频"动作
        load_video_action = QAction('加载视频', self)
        load_video_action.triggered.connect(self.select_video)
        file_menu.addAction(load_video_action)
        
        # 添加分隔线
        file_menu.addSeparator()
        
        # 创建"导出视频"动作
        export_action = QAction('导出视频', self)
        export_action.triggered.connect(self.export_video)
        file_menu.addAction(export_action)
        
        # 添加分隔线
        file_menu.addSeparator()
        
        # 创建"退出"动作
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 处理菜单
        process_menu = menubar.addMenu('处理')
        
        # 创建"处理数据"动作
        process_action = QAction('处理数据', self)
        process_action.triggered.connect(self.process_data)
        process_menu.addAction(process_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        # 添加布局选择子菜单
        layout_menu = QMenu('窗口布局', self)
        view_menu.addMenu(layout_menu)
        
        # 布局选项
        layout_group = QActionGroup(self)
        layout_group.setExclusive(True)
        
        layout_2x2_action = QAction('2×2 布局', self)
        layout_2x2_action.setCheckable(True)
        layout_2x2_action.setChecked(True)  # 默认选中2×2布局
        layout_group.addAction(layout_2x2_action)
        layout_menu.addAction(layout_2x2_action)
        
        layout_1x4_action = QAction('1×4 布局', self)
        layout_1x4_action.setCheckable(True)
        layout_group.addAction(layout_1x4_action)
        layout_menu.addAction(layout_1x4_action)
        
        layout_group.triggered.connect(self.change_layout)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        # 创建"关于"动作
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def add_video_controls(self, toolbar):
        # 选择文件夹按钮
        select_folder_btn = QPushButton('选择文件夹', self)
        select_folder_btn.clicked.connect(self.select_folder)
        toolbar.addWidget(select_folder_btn)
        
        # 选择视频按钮
        select_video_btn = QPushButton('选择视频', self)
        select_video_btn.clicked.connect(self.select_video)
        toolbar.addWidget(select_video_btn)
        
        # 处理按钮
        process_btn = QPushButton('处理', self)
        process_btn.clicked.connect(self.process_data)
        toolbar.addWidget(process_btn)
        
        # 导出视频按钮
        export_btn = QPushButton('导出视频', self)
        export_btn.clicked.connect(self.export_video)
        toolbar.addWidget(export_btn)
        
        # 添加分隔符
        toolbar.addSeparator()
        
        # 重复播放复选框
        self.loop_checkbox = QCheckBox('循环播放', self)
        self.loop_checkbox.setChecked(True)
        toolbar.addWidget(self.loop_checkbox)
        
        # 重新播放按钮
        replay_btn = QPushButton('重新播放', self)
        replay_btn.clicked.connect(self.replay_animations)
        toolbar.addWidget(replay_btn)
        
        # 暂停/开始按钮
        self.play_pause_btn = QPushButton('播放', self)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        toolbar.addWidget(self.play_pause_btn)
        
        # 添加分隔符
        toolbar.addSeparator()
        
        # 帧控制滑动条
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(5, 0, 5, 0)
        
        # 创建帧滑动条
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # 默认最大值，会在加载视频后更新
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        slider_layout.addWidget(self.frame_slider)
        
        # 帧数显示标签
        self.frame_number_label = QLabel("0/0")
        self.frame_number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_number_label.setMinimumWidth(80)
        slider_layout.addWidget(self.frame_number_label)
        
        # 添加滑动条到工具栏
        toolbar.addWidget(slider_container)
    
    def create_main_display_area(self):
        # 创建主分割器（水平方向：左侧显示区域 | 右侧信息面板）
        self.main_horizontal_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_horizontal_splitter.setChildrenCollapsible(False)
        
        # 创建左侧容器
        self.left_container = QWidget()
        left_layout = QVBoxLayout(self.left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建显示标签
        self.recreate_display_labels()
        
        # 创建信息面板
        info_panel = self.create_info_panel()
        
        # 添加左侧容器和信息面板到主分割器
        self.main_horizontal_splitter.addWidget(self.left_container)
        self.main_horizontal_splitter.addWidget(info_panel)
        
        # 设置默认的分割比例
        self.main_horizontal_splitter.setSizes([800, 200])
        
        # 将主分割器添加到主布局
        self.centralWidget().layout().addWidget(self.main_horizontal_splitter)
        
        # 设置默认为2×2布局
        self.set_layout_2x2()
    
    def create_display_label(self, title):
        # 创建帧容器
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.Box)
        frame.setLineWidth(1)
        
        # 使用垂直布局
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 添加标题标签
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(title_label)
        
        # 创建内容标签
        content_label = QLabel()
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_label.setScaledContents(False)  # 不自动缩放内容，我们将手动处理
        content_label.setStyleSheet("background-color: #f0f0f0;")
        content_label.setText(f"{title}内容区域")
        layout.addWidget(content_label)
        
        # 设置为拉伸因子，使内容标签占据大部分空间
        layout.setStretch(0, 0)  # 标题不拉伸
        layout.setStretch(1, 1)  # 内容标签拉伸
        
        # 设置大小策略
        content_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 添加大小调整事件处理
        def resize_event(event):
            if hasattr(content_label, 'pixmap') and content_label.pixmap() and not content_label.pixmap().isNull():
                # 为标签大小留出一些边距
                label_size = event.size()
                scaled_pixmap = content_label.pixmap().scaled(
                    label_size.width() - 10, 
                    label_size.height() - 10,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                content_label.setPixmap(scaled_pixmap)
            QLabel.resizeEvent(content_label, event)
        
        content_label.resizeEvent = resize_event
        
        # 保存内容标签的引用
        frame.content_label = content_label
        
        # 根据标题保存标签引用
        if title == "气泡流图像":
            self.content_labels['flow'] = content_label
        elif title == "气泡实时检测":
            self.content_labels['detection'] = content_label
        elif title == "简化气泡流场":
            self.content_labels['3d'] = content_label
        elif title == "气泡流场重建":
            self.content_labels['empty'] = content_label
        
        # 设置最小大小
        frame.setMinimumSize(200, 200)
        
        # 保存标题以便后续引用
        frame.title = title
        
        return frame
    
    def recreate_display_labels(self):
        self.labelFlow = self.create_display_label("气泡流图像")
        self.labelDetection = self.create_display_label("气泡实时检测")
        self.label3D = self.create_display_label("简化气泡流场")
        self.labelEmpty = self.create_display_label("气泡流场重建")
    
    def create_info_panel(self):
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(15)
        
        # 创建信息显示组
        info_group = QGroupBox("气泡信息")
        info_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        info_group_layout = QVBoxLayout(info_group)
        
        # 创建信息标签
        self.info_label = QLabel("这里将显示气泡信息")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 10px;
                font-family: Consolas, Monaco, monospace;
                font-size: 16px;
                min-height: 100px;
            }
        """)
        self.info_label.setWordWrap(True)
        info_group_layout.addWidget(self.info_label)
        
        # 创建日志组
        log_group = QGroupBox("日志")
        log_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        log_group_layout = QVBoxLayout(log_group)
        
        # 创建日志标签
        self.log_label = QLabel()
        self.log_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.log_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 10px;
                font-family: Consolas, Monaco, monospace;
                font-size: 16px;
                min-height: 100px;
            }
        """)
        self.log_label.setWordWrap(True)
        log_group_layout.addWidget(self.log_label)
        
        # 添加组到面板
        info_layout.addWidget(info_group, 1)  # 设置拉伸因子为1
        info_layout.addWidget(log_group, 1)   # 设置拉伸因子为1
        
        return info_panel
    
    def change_layout(self, action):
        layout_text = action.text()
        self.add_log(f"切换布局为: {layout_text}")
        
        if layout_text == '2×2 布局':
            self.set_layout_2x2()
        elif layout_text == '1×4 布局':
            self.set_layout_1x4()
    
    def set_layout_2x2(self):
        try:
            # 清空当前布局中的所有部件
            for i in reversed(range(self.left_container.layout().count())):
                item = self.left_container.layout().itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            
            # 重新创建标签
            self.recreate_display_labels()
            
            # 创建嵌套的分割器结构
            # 主水平分割器
            self.splitter_main = QSplitter(Qt.Orientation.Horizontal)
            self.splitter_main.setChildrenCollapsible(False)
            
            # 左侧垂直分割器
            self.splitter_left = QSplitter(Qt.Orientation.Vertical)
            self.splitter_left.setChildrenCollapsible(False)
            self.splitter_left.addWidget(self.labelFlow)
            self.splitter_left.addWidget(self.labelDetection)
            
            # 右侧垂直分割器
            self.splitter_right = QSplitter(Qt.Orientation.Vertical)
            self.splitter_right.setChildrenCollapsible(False)
            self.splitter_right.addWidget(self.label3D)
            self.splitter_right.addWidget(self.labelEmpty)
            
            # 将左右分割器添加到主分割器
            self.splitter_main.addWidget(self.splitter_left)
            self.splitter_main.addWidget(self.splitter_right)
            
            # 设置初始大小
            self.splitter_left.setSizes([300, 300])
            self.splitter_right.setSizes([300, 300])
            self.splitter_main.setSizes([500, 500])
            
            # 将主分割器添加到左侧容器
            self.left_container.layout().addWidget(self.splitter_main)
        except Exception as e:
            print(f"设置2×2布局时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def set_layout_1x4(self):
        try:
            # 清空当前布局中的所有部件
            for i in reversed(range(self.left_container.layout().count())):
                item = self.left_container.layout().itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            
            # 重新创建标签
            self.recreate_display_labels()
            
            # 创建水平分割器
            self.splitter_main = QSplitter(Qt.Orientation.Horizontal)
            self.splitter_main.setChildrenCollapsible(False)
            
            # 添加所有显示标签到水平分割器
            self.splitter_main.addWidget(self.labelFlow)
            self.splitter_main.addWidget(self.labelDetection)
            self.splitter_main.addWidget(self.label3D)
            self.splitter_main.addWidget(self.labelEmpty)
            
            # 设置初始大小平均分配
            total_width = self.left_container.width()
            widget_width = total_width // 4
            self.splitter_main.setSizes([widget_width, widget_width, widget_width, widget_width])
            
            # 将主分割器添加到左侧容器
            self.left_container.layout().addWidget(self.splitter_main)
        except Exception as e:
            print(f"设置1×4布局时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_history.append(log_entry)
        
        # 只保留最近的日志（例如最近20条）
        if len(self.log_history) > 20:
            self.log_history = self.log_history[-20:]
        
        # 更新日志显示
        log_text = "\n".join(self.log_history)
        if hasattr(self, 'log_label'):
            self.log_label.setText(log_text)
        
        # 打印到控制台（调试用）
        print(log_entry)
    
    def select_folder(self):
        self.add_log("选择文件夹")
        self.statusbar.showMessage("已选择文件夹")
    
    def select_video(self):
        self.add_log("选择视频")
        self.statusbar.showMessage("已选择视频")
    
    def process_data(self):
        self.add_log("处理数据")
        self.statusbar.showMessage("数据处理完成")
        
        # 创建一些测试帧
        self.frames = [QImage(600, 400, QImage.Format.Format_RGB32) for _ in range(10)]
        for i, img in enumerate(self.frames):
            img.fill(QColor(100, 150, 200))  # 蓝色背景
            # 画一个移动的圆形
            center_x = 100 + i * 40
            for r in range(50, 0, -1):
                for y in range(max(150-r, 0), min(150+r, 400)):
                    for x in range(max(center_x-r, 0), min(center_x+r, 600)):
                        if (x - center_x)**2 + (y - 150)**2 <= r**2:
                            img.setPixelColor(x, y, QColor(255, 255, 0))  # 黄色圆形
        
        # 更新滑动条
        self.frame_slider.setMaximum(len(self.frames) - 1)
        self.frame_slider.setEnabled(True)
        
        # 显示第一帧
        self.update_display(0)
    
    def export_video(self):
        self.add_log("导出视频")
        self.statusbar.showMessage("视频导出完成")
    
    def show_about(self):
        QMessageBox.about(self, "关于", 
                          "简化测试程序\n\n"
                          "用于测试窗口布局和基本功能的程序")
    
    def toggle_play_pause(self):
        if not self.frames:
            return
            
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        if not self.frames or self.is_playing:
            return
            
        # 创建帧定时器
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.next_frame)
        self.frame_timer.start(100)  # 100毫秒切换一帧
        
        self.is_playing = True
        self.play_pause_btn.setText("暂停")
        self.add_log("开始播放")
    
    def pause_playback(self):
        if not self.is_playing:
            return
            
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()
            
        self.is_playing = False
        self.play_pause_btn.setText("播放")
        self.add_log("暂停播放")
    
    def next_frame(self):
        if not self.frames:
            return
            
        # 计算下一帧索引
        next_index = self.current_frame_index + 1
        
        # 检查是否到达视频末尾
        if next_index >= len(self.frames):
            if self.loop_checkbox.isChecked():
                # 循环播放，回到第一帧
                next_index = 0
            else:
                # 停止播放
                self.pause_playback()
                return
        
        # 更新当前帧索引
        self.current_frame_index = next_index
        
        # 更新滑动条位置（不触发valueChanged信号）
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_index)
        self.frame_slider.blockSignals(False)
        
        # 更新显示
        self.update_display(next_index)
    
    def replay_animations(self):
        if not self.frames:
            return
            
        self.current_frame_index = 0
        self.frame_slider.setValue(0)
        self.update_display(0)
        self.add_log("重新播放")
        
        if not self.is_playing:
            self.start_playback()
    
    def frame_slider_changed(self, value):
        if not self.frames:
            return
            
        self.current_frame_index = value
        self.update_display(value)
    
    def update_display(self, frame_index):
        if not self.frames or frame_index >= len(self.frames):
            return
        
        # 更新帧号显示
        self.frame_number_label.setText(f"{frame_index + 1}/{len(self.frames)}")
        
        # 获取当前帧
        frame = self.frames[frame_index]
        
        # 创建QPixmap并设置到标签
        pixmap = QPixmap.fromImage(frame)
        
        # 更新主图像显示
        if self.content_labels['flow']:
            # 根据标签大小缩放pixmap
            label_size = self.content_labels['flow'].size()
            scaled_pixmap = pixmap.scaled(
                label_size.width() - 10, 
                label_size.height() - 10,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.content_labels['flow'].setPixmap(scaled_pixmap)
        
        # 更新检测图像显示（稍微修改一下颜色）
        if self.content_labels['detection']:
            # 创建检测图像的副本并修改
            det_img = frame.copy()
            for y in range(det_img.height()):
                for x in range(det_img.width()):
                    color = det_img.pixelColor(x, y)
                    if color.red() > 200 and color.green() > 200:  # 黄色区域
                        # 改为红色
                        det_img.setPixelColor(x, y, QColor(255, 0, 0))
            
            det_pixmap = QPixmap.fromImage(det_img)
            label_size = self.content_labels['detection'].size()
            scaled_det_pixmap = det_pixmap.scaled(
                label_size.width() - 10, 
                label_size.height() - 10,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.content_labels['detection'].setPixmap(scaled_det_pixmap)
            
        # 更新3D标签显示
        if self.content_labels['3d']:
            self.content_labels['3d'].setText(f"3D视图 - 帧 {frame_index + 1}")
        
        # 更新重建标签显示
        if self.content_labels['empty']:
            self.content_labels['empty'].setText(f"重建视图 - 帧 {frame_index + 1}")

def main():
    app = QApplication(sys.argv)
    window = SimpleTestWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
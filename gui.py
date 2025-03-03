import os
import time
from PyQt6.QtWidgets import (QMainWindow, QFileDialog, QPushButton, QLabel, 
                            QVBoxLayout, QWidget, QHBoxLayout, QProgressBar, 
                            QMessageBox, QMenuBar, QMenu, QStatusBar,
                            QFrame, QSizePolicy, QCheckBox, QGridLayout,
                            QScrollArea, QColorDialog, QSlider, QApplication,
                            QGroupBox, QSplitter, QToolBar)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import (QMovie, QPalette, QColor, QFont, QPixmap, QImage, 
                        QIcon, QAction, QActionGroup)
import cv2
import imageio
import numpy as np

from image_processor import process_images
from video_processor import process_video, get_video_info
from animation_exporter import export_animation
from ellipse_fitting import draw_ellipse_on_image, draw_bounding_box_on_image, draw_rotated_rectangle
from reconstruction3d import reconstruct_3d

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BFRM - 气泡流检测与重构")
        self.setGeometry(100, 100, 1200, 800)
        
        # YOLO模型相关
        self.model = None
        self.trajectories = {}  # 存储气泡轨迹
        self.bubble_info = {}  # 存储气泡信息
        
        # 使用相对路径设置图标
        icon_path = os.path.join(os.getcwd(), "bubble.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        # 设置默认路径为当前目录
        self.default_folder = os.getcwd()
        self.result_root = os.path.join(os.getcwd(), "results")
        self.output_folder = self.create_result_folder()
        self.data_folder = ""  # 初始化数据文件夹路径
        
        # 初始化临时GIF文件字典
        self.temp_gif_files = {}
        
        # 存储内容标签的引用
        self.content_labels = {
            'flow': None,
            'detection': None,
            '3d': None,
            'info': None
        }
        
        self.frames = []
        self.frames_bbox = []
        self.frames_ellipse = []
        self.movies = {}  # 存储不同类型的QMovie对象
        self.current_labels = {}  # 存储不同类型的显示标签
        self.current_frame_index = 0  # 当前帧索引
        self.bubble_trail = []  # 存储气泡轨迹
        self.playback_speed = 1.0  # 播放速度倍率
        self.max_speed = 0  # 添加最大速度属性
        self.log_history = []  # 存储日志历史
        self.log_update_interval = 5  # 每5帧更新一次日志
        self.frame_info_dict = {}  # 存储每一帧的信息字典
        
        # 添加视频处理相关变量
        self.video_path = ""
        self.is_video_mode = False
        self.video_fps = 30  # 默认帧率
        self.is_playing = False  # 视频是否在播放
        
        # 添加视频帧显示定时器
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.next_frame)
        
        # 添加3D可视化组件
        self.canvas_3d = None
        
        self.init_ui()
        
        self.movies = {
            'flow': QMovie(),
            'bbox': QMovie()
        }
        
        # 添加程序启动日志
        self.add_log("程序启动完成")
        self.add_log("提示: 可以通过'选择视频'按钮直接加载视频文件")
        self.add_log("提示: 视频加载后可以使用滑动条控制播放位置")
        self.add_log("提示: 可以使用'导出视频'按钮将处理后的视频保存")
    def create_result_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self.result_root, timestamp)
        os.makedirs(folder, exist_ok=True)
        return folder
    def init_ui(self):
        """初始化UI界面"""
        # 设置窗口标题
        self.setWindowTitle("气泡流检测与重构 v1.0")
        self.resize(1600, 900)  # 设置初始窗口大小

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
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
        
    def create_main_display_area(self):
        """创建主显示区域，使用分割器实现可调整大小的布局"""
        # 创建主分割器（水平方向：左侧显示区域 | 右侧信息面板）
        self.main_horizontal_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_horizontal_splitter.setChildrenCollapsible(False)
        
        # 创建左侧容器（用于包含显示区域）
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
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.main_horizontal_splitter)
        
        # 将滚动区域添加到主布局
        self.centralWidget().layout().addWidget(scroll_area)
        
        # 设置默认为2×2布局
        self.set_layout_2x2()
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        # 创建"加载视频"动作
        load_video_action = QAction('加载视频', self)
        load_video_action.triggered.connect(self.select_video)
        file_menu.addAction(load_video_action)
        
        # 添加分隔线
        file_menu.addSeparator()
        
        # 创建"导出视频"动作
        export_action = QAction('导出视频', self)
        export_action.triggered.connect(self.export_video)
        export_action.setEnabled(False)  # 初始禁用
        self.export_action = export_action  # 保存引用以便后续启用
        file_menu.addAction(export_action)
        
        # 添加分隔线
        file_menu.addSeparator()
        
        # 创建"退出"动作
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 处理菜单
        process_menu = menubar.addMenu('处理')
        
        # 创建"重新加载视频"动作
        reload_action = QAction('重新加载视频', self)
        reload_action.triggered.connect(self.select_video)
        process_menu.addAction(reload_action)
        
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
        
        # 添加播放速度子菜单
        speed_menu = QMenu('播放速度', self)
        view_menu.addMenu(speed_menu)
        
        # 播放速度选项
        speed_group = QActionGroup(self)
        speed_group.setExclusive(True)
        
        # 添加不同的播放速度选项
        speeds = ['0.25x', '0.5x', '1.0x', '1.5x', '2.0x', '4.0x']
        for speed in speeds:
            speed_action = QAction(speed, self)
            speed_action.setCheckable(True)
            if speed == '1.0x':  # 默认选中1.0x速度
                speed_action.setChecked(True)
            # 添加快捷键
            if speed == '1.0x':
                speed_action.setShortcut('Ctrl+1')
            elif speed == '2.0x':
                speed_action.setShortcut('Ctrl+2')
            elif speed == '4.0x':
                speed_action.setShortcut('Ctrl+4')
            elif speed == '0.5x':
                speed_action.setShortcut('Ctrl+5')
            speed_group.addAction(speed_action)
            speed_menu.addAction(speed_action)
        
        speed_group.triggered.connect(self.change_playback_speed)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        # 创建"关于"动作
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    def add_video_controls(self, toolbar):
        """添加视频控制按钮到工具栏"""
        # 选择视频按钮
        select_video_btn = QPushButton('选择视频', self)
        select_video_btn.setIcon(QIcon("icons/video.png"))
        select_video_btn.clicked.connect(self.select_video)
        toolbar.addWidget(select_video_btn)
        
        # 重新加载按钮(原处理按钮)
        reload_btn = QPushButton('重新加载', self)
        reload_btn.setIcon(QIcon("icons/reload.png"))
        reload_btn.clicked.connect(self.select_video)
        toolbar.addWidget(reload_btn)
        
        # 导出视频按钮
        export_btn = QPushButton('导出视频', self)
        export_btn.setIcon(QIcon("icons/export.png"))
        export_btn.clicked.connect(self.export_video)
        export_btn.setEnabled(False)  # 初始禁用
        self.export_btn = export_btn  # 保存引用以便后续启用
        toolbar.addWidget(export_btn)
        
        # 添加分隔符
        toolbar.addSeparator()
        
        # 播放速度控制
        speed_menu = QMenu()
        speeds = ['0.25x', '0.5x', '1.0x', '1.5x', '2.0x', '4.0x']
        self.speed_actions = {}
        
        for speed in speeds:
            action = QAction(speed, self)
            action.setCheckable(True)
            if speed == '1.0x':
                action.setChecked(True)
            # 添加快捷键
            if speed == '1.0x':
                action.setShortcut('Ctrl+1')
            elif speed == '2.0x':
                action.setShortcut('Ctrl+2')
            elif speed == '4.0x':
                action.setShortcut('Ctrl+4')
            elif speed == '0.5x':
                action.setShortcut('Ctrl+5')
            action.triggered.connect(lambda checked, s=speed: self.change_playback_speed(action=s))
            speed_menu.addAction(action)
            self.speed_actions[speed] = action
        
        speed_btn = QPushButton('速度: 1.0x', self)
        speed_btn.setIcon(QIcon("icons/speed.png"))  # 添加速度图标
        speed_btn.setMenu(speed_menu)
        self.speed_btn = speed_btn  # 保存引用以便后续更新
        toolbar.addWidget(speed_btn)
        
        # 重复播放复选框
        self.loop_checkbox = QCheckBox('循环播放', self)
        self.loop_checkbox.setChecked(True)
        self.loop_checkbox.clicked.connect(self.toggle_loop)
        toolbar.addWidget(self.loop_checkbox)
        
        # 重新播放按钮
        replay_btn = QPushButton('重新播放', self)
        replay_btn.setIcon(QIcon("icons/replay.png"))
        replay_btn.clicked.connect(self.replay_animations)
        toolbar.addWidget(replay_btn)
        
        # 暂停/开始按钮
        self.play_pause_btn = QPushButton('暂停', self)
        self.play_pause_btn.setIcon(QIcon("icons/pause.png"))
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
        self.frame_slider.setEnabled(False)  # 默认禁用
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        slider_layout.addWidget(self.frame_slider)
        
        # 帧数显示标签
        self.frame_number_label = QLabel("0/0")
        self.frame_number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_number_label.setMinimumWidth(80)
        slider_layout.addWidget(self.frame_number_label)
        
        # 添加滑动条到工具栏
        toolbar.addWidget(slider_container)
    def change_title_font_size(self, action):
        """更改标题字体大小"""
        size_map = {
            '小 (18px)': 18,
            '中 (22px)': 22,
            '大 (26px)': 26,
            '较大 (30px)': 30
        }
        
        size = size_map[action.text()]
        
        # 更新所有标题标签的字体大小
        title_labels = [
            self.labelFlow.layout().itemAt(0).widget(),
            self.labelDetection.layout().itemAt(0).widget(),
            self.label3D.layout().itemAt(0).widget(),
            self.labelEmpty.layout().itemAt(0).widget()
        ]
        
        for label in title_labels:
            title_font = label.font()
            title_font.setPointSize(size)
            label.setFont(title_font)
            label.setStyleSheet(f"""
                QLabel {{
                    font-size: {size}px;
                    font-weight: bold;
                    color: #333333;
                    padding: 8px;
                }}
            """)
        
        self.add_log(f"更改标题字体大小为: {action.text()}")

    def change_content_font_size(self, action):
        """更改内容字体大小"""
        size_map = {
            '小 (14px)': 14,
            '中 (16px)': 16,
            '大 (18px)': 18,
            '较大 (20px)': 20
        }
        
        size = size_map[action.text()]
        
        # 更新信息显示和日志标签的字体大小
        if 'info' in self.content_labels and self.content_labels['info'] is not None:
            self.content_labels['info'].setStyleSheet(f"""
                QLabel {{
                    background-color: white;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 10px;
                    font-family: Consolas, Monaco, monospace;
                    font-size: {size}px;
                    min-height: 100px;
                }}
            """)
        
        if hasattr(self, 'log_label'):
            self.log_label.setStyleSheet(f"""
                QLabel {{
                    background-color: white;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 10px;
                    font-family: Consolas, Monaco, monospace;
                    font-size: {size}px;
                    min-height: 100px;
                }}
            """)
        
        self.add_log(f"更改内容字体大小为: {action.text()}")
    def create_display_label(self, title):
        """创建用于显示内容的标签"""
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
        
        # 根据不同的标题创建不同的内容
        if title == "简化气泡流场":
            # 创建3D画布
            try:
                self.canvas_3d = reconstruct_3d(None, None)
                self.canvas_3d.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.canvas_3d.setMinimumSize(50, 50)
                layout.addWidget(self.canvas_3d, 1)
                self.content_labels['3d'] = self.canvas_3d
            except Exception as e:
                # 如果创建3D画布失败，使用普通标签替代
                self.add_log(f"创建3D画布失败: {str(e)}")
                content_label = QLabel("3D视图")
                content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                content_label.setScaledContents(False)
                layout.addWidget(content_label, 1)
                self.content_labels['3d'] = content_label
        else:
            # 创建内容标签
            content_label = QLabel()
            content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content_label.setScaledContents(False)  # 不自动缩放内容，我们将手动处理
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
            elif title == "气泡流场重建":
                self.content_labels['empty'] = content_label
        
        # 设置最小大小
        frame.setMinimumSize(200, 200)
        
        # 保存标题以便后续引用
        frame.title = title
        
        return frame
    
    def recreate_display_labels(self):
        """重新创建所有显示标签"""
        self.labelFlow = self.create_display_label("气泡流图像")
        self.labelDetection = self.create_display_label("气泡实时检测")
        self.label3D = self.create_display_label("简化气泡流场")
        self.labelEmpty = self.create_display_label("气泡流场重建")
    def reset_processing(self):
        """重置处理状态"""
        self.frames = []
        self.frames_bbox = []
        self.frames_ellipse = []
        self.progress_bar.setValue(0)
        
        # 停止所有动画
        for movie in self.movies.values():
            if movie:
                movie.stop()
        
        # 清空显示
        for label in self.content_labels.values():
            if label is not None:
                label.clear()
            
        self.statusbar.showMessage('已重置')
        self.output_folder = self.create_result_folder()
        self.add_log("重置处理状态")
        
    def select_folder(self):
        """此功能已被禁用"""
        self.add_log("文件夹处理功能已被禁用")
        self.statusbar.showMessage("文件夹处理功能已被禁用")
        QMessageBox.information(self, "功能已禁用", "文件夹处理功能已被禁用，请使用视频处理功能")
            
    def select_video(self):
        """选择并加载视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            self.default_folder,
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)"
        )
        
        if file_path:
            self.data_folder = os.path.dirname(file_path)
            self.video_path = file_path
            self.is_video_mode = True
            
            # 重置状态
            self.reset_processing()
            
            # 初始化YOLO模型
            if not self.init_yolo_model():
                QMessageBox.warning(self, "警告", "YOLO模型初始化失败，将无法进行气泡检测！")
            
            # 加载视频
            self.load_video()
            
    def load_video(self):
        """加载视频文件"""
        if not self.video_path or not os.path.exists(self.video_path):
            return
            
        try:
            self.statusbar.showMessage("正在加载视频...")
            self.add_log(f"开始加载视频: {self.video_path}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 获取视频信息
            self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 更新进度条
            self.progress_bar.setMaximum(total_frames)
            self.progress_bar.setValue(0)
            
            # 清空帧列表
            self.frames = []
            self.trajectories = {}
            
            # 读取视频帧
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.frames.append(frame)
                frame_count += 1
                
                # 更新进度
                self.progress_bar.setValue(frame_count)
                if frame_count % 10 == 0:
                    self.statusbar.showMessage(f"正在加载视频... {frame_count}/{total_frames}")
                    QApplication.processEvents()
            
            # 关闭视频文件
            cap.release()
            
            # 设置滑动条范围
            self.frame_slider.setMaximum(len(self.frames) - 1)
            self.frame_slider.setValue(0)
            
            # 更新显示
            self.update_display(0)
            
            # 启用导出按钮
            self.export_action.setEnabled(True)
            
            self.statusbar.showMessage(f"视频加载完成，共 {len(self.frames)} 帧")
            self.add_log(f"视频加载完成: {len(self.frames)} 帧, {self.video_fps} FPS")
            
            # 自动开始播放
            self.start_playback()
            
        except Exception as e:
            self.statusbar.showMessage("视频加载失败")
            error_msg = f"加载视频时出错: {str(e)}"
            self.add_log(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            import traceback
            traceback.print_exc()

    def process_data(self):
        """处理选定的视频数据"""
        # 检查是否已选择视频
        if not self.data_folder or not self.is_video_mode:
            self.statusbar.showMessage('请先选择视频文件')
            QMessageBox.warning(self, "错误", "请先选择视频文件！")
            return
            
        try:
            # 启动处理线程
            self.add_log("开始处理视频数据...")
            
            # 视频模式处理
            self.process_video_data()
                
        except Exception as e:
            self.statusbar.showMessage("处理出错")
            error_msg = f"处理数据时出错: {str(e)}"
            self.add_log(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            import traceback
            traceback.print_exc()
            
    def process_video_data(self):
        """处理视频数据"""
        try:
            self.statusbar.showMessage("开始加载视频文件")
            self.add_log(f"开始加载视频文件: {self.video_file}")

            # 更新界面，重置进度条
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # 确保进度条重置立即显示
            
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_file)
            if not cap.isOpened():
                raise ValueError(f"无法读取视频: {self.video_file}")
            
            # 获取视频信息
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"视频总帧数: {self.frame_count}, FPS: {self.video_fps}")
            
            # 初始化帧列表和帧信息
            self.frames = []
            self.detection_frames = []
            self.frame_info_dict = {}
            
            # 记录播放状态变量
            self.playback_started = False
            was_playing = self.is_playing
            
            # 如果当前正在播放，先暂停(但不更新UI)
            if was_playing:
                self.frame_timer.stop()
                self.is_playing = False
            
            # 初始化帧索引和帧信息
            self.current_frame_index = 0
            self.frame_slider.setMaximum(self.frame_count - 1)
            self.frame_slider.setEnabled(True)
            
            # 设置为视频模式
            self.is_video_mode = True
            
            # 开始加载帧
            frames_before_play = min(30, self.frame_count // 10)  # 加载大约30帧或10%的帧后开始播放
            
            # 读取视频帧
            frames_read = 0
            for i in range(self.frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 添加帧
                self.frames.append(frame)
                self.detection_frames.append(frame.copy())  # 简单复制，不做实际检测
                self.frame_info_dict[i] = {'frame': i, 'bubbles': []}
                frames_read += 1
                
                # 更新进度 - 确保基于实际预期的总帧数计算
                progress = min(int(100 * frames_read / self.frame_count), 100)  # 确保不超过100%
                self.progress_bar.setValue(progress)
                
                # 在加载一定数量帧后开始播放（如果尚未开始）
                if not self.playback_started and frames_read >= frames_before_play:
                    self.update_interface_display()
                    self.start_playback()
                    self.playback_started = True
                    self.statusbar.showMessage(f"已加载 {frames_read} 帧，正在继续加载...")
                    self.add_log(f"开始播放已加载的帧: {frames_read} 帧")
                
                # 每10帧更新一次UI，减少UI更新次数，提高加载速度
                if frames_read % 10 == 0 or frames_read < 10:
                    QApplication.processEvents()
            
            # 释放视频对象
            cap.release()
            
            # 确保进度条显示正确的最终状态 - 基于实际读取的帧数
            actual_progress = 100  # 已完成加载
            self.progress_bar.setValue(actual_progress)
            QApplication.processEvents()  # 确保进度条更新立即显示
            
            # 保存当前播放状态和位置
            currently_playing = self.is_playing
            current_pos = self.current_frame_index
            
            # 如果当前正在播放，先暂停(但不更新UI)，以便更新界面
            if currently_playing:
                self.frame_timer.stop()
                self.is_playing = False
            
            # 更新最终状态
            self.statusbar.showMessage(f"视频加载完成: {len(self.frames)}帧, {self.video_fps} FPS")
            self.add_log(f"加载完成: {len(self.frames)} 帧, {self.video_fps} FPS")
            
            # 先保存播放状态，然后更新界面
            self.update_interface_display()
            
            # 恢复当前帧索引和滑动条位置
            self.current_frame_index = current_pos
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(current_pos)
            self.frame_slider.blockSignals(False)
            
            # 更新当前帧显示
            self.update_display(current_pos)
            self.add_log("更新界面显示")
            
            # 如果之前在播放，恢复播放状态
            if currently_playing:
                self.is_playing = True
                interval = int(1000 / (self.video_fps * self.playback_speed))
                self.frame_timer.start(interval)
            
            self.add_log("视频加载完成")
            
        except Exception as e:
            self.statusbar.showMessage("处理视频失败")
            self.add_log(f"处理视频失败: {str(e)}")
            print(str(e))
            import traceback
            traceback.print_exc()
    
    def process_folder_data(self):
        """文件夹处理功能已被禁用"""
        self.add_log("文件夹处理功能已被禁用")
        self.statusbar.showMessage("文件夹处理功能已被禁用")
        QMessageBox.information(self, "功能已禁用", "文件夹处理功能已被禁用，请使用视频处理功能")
    
    def generate_bubble_flow_images(self):
        """生成气泡流图像序列"""
        self.statusbar.showMessage("正在处理气泡图像...")
        
        # 处理逻辑
        return
        
    def generate_bubble_detection_images(self):
        """生成气泡检测图像序列"""
        try:
            # 异常处理
            if not self.frames or len(self.frames) == 0:
                self.statusbar.showMessage("处理出错：没有找到帧")
                return False
            
            self.statusbar.showMessage("正在生成气泡检测图像...")
            self.progress_bar.setValue(0)
            self.add_log("开始生成气泡检测图像")
            
            # 创建气泡检测图像序列
            self.frames_bbox = []
            total_frames = len(self.frames)
            
            for i, frame in enumerate(self.frames):
                # 创建原始帧的副本
                detection_frame = frame.copy()
                
                # 如果有帧信息，绘制气泡检测框
                if i in self.frame_info_dict and 'bubbles' in self.frame_info_dict[i]:
                    bubbles = self.frame_info_dict[i]['bubbles']
                    for bubble in bubbles:
                        if 'x' in bubble and 'y' in bubble and 'radius' in bubble:
                            # 绘制圆形检测框
                            cv2.circle(
                                detection_frame,
                                (int(bubble['x']), int(bubble['y'])),
                                int(bubble['radius']),
                                (0, 0, 255),  # 红色边框
                                2  # 线宽
                            )
                            
                            # 添加气泡ID或其他信息
                            if 'id' in bubble:
                                cv2.putText(
                                    detection_frame,
                                    f"ID: {bubble['id']}",
                                    (int(bubble['x']) - 20, int(bubble['y']) - int(bubble['radius']) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    1
                                )
                
                # 添加到检测图像序列
                self.frames_bbox.append(detection_frame)
                
                # 更新进度
                progress = int(100 * (i + 1) / total_frames)
                self.progress_bar.setValue(progress)
                if (i + 1) % 20 == 0 or i == total_frames - 1:
                    self.statusbar.showMessage(f"正在生成气泡检测图像... {progress}%")
                    QApplication.processEvents()
            
            self.statusbar.showMessage("气泡检测图像生成完成")
            self.add_log(f"生成了 {len(self.frames_bbox)} 个气泡检测图像")
            return True
            
        except Exception as e:
            self.statusbar.showMessage("处理出错")
            error_msg = f"生成气泡检测图像时出错: {str(e)}"
            self.add_log(error_msg)
            print(error_msg)
            return False
        
    def preprocess_frame_info(self):
        """预处理帧信息"""
        try:
            self.statusbar.showMessage("正在预处理帧信息...")
            self.add_log("开始预处理帧信息")
            
            # 如果帧信息字典为空，创建默认信息
            if not self.frame_info_dict:
                self.frame_info_dict = {}
                for i in range(len(self.frames)):
                    self.frame_info_dict[i] = {
                        'frame_idx': i,
                        'bubbles': []
                    }
                    
                    # 创建一些示例气泡（在实际应用中，这里会使用真实检测结果）
                    bubble_count = np.random.randint(2, 6)  # 随机气泡数量
                    for b in range(bubble_count):
                        height, width = self.frames[i].shape[:2]
                        bubble = {
                            'id': b,
                            'x': np.random.randint(100, width - 100),
                            'y': np.random.randint(100, height - 100),
                            'radius': np.random.randint(15, 35),
                            'area': 0,  # 将在后续计算
                            'velocity': np.random.uniform(1.0, 5.0)
                        }
                        # 计算面积
                        bubble['area'] = np.pi * (bubble['radius'] ** 2)
                        self.frame_info_dict[i]['bubbles'].append(bubble)
            
            # 计算一些统计信息
            total_bubbles = sum(len(info.get('bubbles', [])) for info in self.frame_info_dict.values())
            avg_bubbles_per_frame = total_bubbles / len(self.frame_info_dict) if self.frame_info_dict else 0
            
            self.add_log(f"帧信息预处理完成: 共 {len(self.frame_info_dict)} 帧，平均每帧 {avg_bubbles_per_frame:.1f} 个气泡")
            self.statusbar.showMessage("帧信息预处理完成")
            
            return True
            
        except Exception as e:
            self.statusbar.showMessage("处理出错")
            error_msg = f"预处理帧信息时出错: {str(e)}"
            self.add_log(error_msg)
            print(error_msg)
            return False
        
    def export_animation_files(self):
        """准备动画显示"""
        try:
            self.statusbar.showMessage("正在准备动画显示...")
            self.add_log("开始准备动画显示")
            
            # 检查是否有帧可以显示
            if not self.frames or len(self.frames) == 0:
                self.statusbar.showMessage("准备失败：没有可用帧")
                self.add_log("准备动画失败：没有可用帧")
                return False
            
            # 对于流场图像，直接使用原始帧
            flow_frames = self.frames
            
            # 对于检测图像，如果没有生成，使用原始帧
            if not self.frames_bbox or len(self.frames_bbox) == 0:
                self.frames_bbox = [frame.copy() for frame in self.frames]
            
            # 直接使用帧序列进行显示，不导出GIF
            self.add_log("视频模式：直接使用帧序列显示")
            
            # 显示第一帧
            if self.content_labels['flow'] and len(flow_frames) > 0:
                first_frame = flow_frames[0]
                rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # 缩放图像以适应标签大小
                label_size = self.content_labels['flow'].size()
                scaled_pixmap = pixmap.scaled(
                    label_size.width() - 10, 
                    label_size.height() - 10,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.content_labels['flow'].setPixmap(scaled_pixmap)
            
            # 显示第一帧的检测图像
            if self.content_labels['detection'] and len(self.frames_bbox) > 0:
                first_bbox_frame = self.frames_bbox[0]
                rgb_bbox = cv2.cvtColor(first_bbox_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_bbox.shape
                bytes_per_line = ch * w
                q_img_bbox = QImage(rgb_bbox.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap_bbox = QPixmap.fromImage(q_img_bbox)
                
                # 缩放边界框图像
                if self.content_labels['detection'] and pixmap_bbox:
                    label_size = self.content_labels['detection'].size()
                    scaled_pixmap_bbox = pixmap_bbox.scaled(
                        label_size.width() - 10, 
                        label_size.height() - 10,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.content_labels['detection'].setPixmap(scaled_pixmap_bbox)
            
            self.statusbar.showMessage("帧序列准备完成")
            self.add_log("帧序列准备完成，可以开始播放")
            return True
                
        except Exception as e:
            self.statusbar.showMessage("处理出错")
            error_msg = f"准备动画显示时出错: {str(e)}"
            self.add_log(error_msg)
            print(error_msg)
            return False
        
    def update_interface_display(self):
        """更新界面显示"""
        try:
            self.statusbar.showMessage("正在更新界面显示...")
            self.add_log("更新界面显示")
            
            # 显示当前帧或第一帧（如果没有当前帧）
            if self.frames and len(self.frames) > 0:
                # 确保帧滑动条正确设置
                self.frame_slider.setMaximum(len(self.frames) - 1)
                
                # 保存当前播放状态
                was_playing = self.is_playing
                current_frame = self.current_frame_index
                
                # 仅当帧索引无效时初始化为0，否则保持当前位置
                if current_frame < 0 or current_frame >= len(self.frames):
                    self.current_frame_index = 0
                    self.frame_slider.setValue(0)
                    self.update_display(0)
                else:
                    # 保持当前帧索引
                    self.frame_slider.setValue(current_frame)
                    self.update_display(current_frame)
                
                self.frame_slider.setEnabled(True)
                
                # 更新帧数标签
                self.frame_number_label.setText(f"{self.current_frame_index + 1}/{len(self.frames)}")
                
                # 如果之前在播放，确保继续播放
                if was_playing and not self.is_playing:
                    self.is_playing = True
                    interval = int(1000 / (self.video_fps * self.playback_speed))
                    self.frame_timer.start(interval)
            
            # 更新信息标签 - 显示处理摘要
            summary_text = "<div style='font-family: monospace;'>"
            summary_text += f"<b>处理摘要</b><br>"
            
            if self.is_video_mode:
                summary_text += f"视频文件: {os.path.basename(self.video_file)}<br>"
                summary_text += f"帧数: {len(self.frames)}<br>"
                summary_text += f"帧率: {self.video_fps} FPS<br>"
            else:
                summary_text += f"图像文件夹: {os.path.basename(self.data_folder)}<br>"
                summary_text += f"图像数量: {len(self.frames)}<br>"
                summary_text += f"CSV文件: {os.path.basename(self.csv_file)}<br>"
            
            # 添加气泡统计信息
            total_bubbles = sum(len(info.get('bubbles', [])) for info in self.frame_info_dict.values())
            if self.frame_info_dict:
                avg_bubbles = total_bubbles / len(self.frame_info_dict)
                summary_text += f"检测到气泡: {total_bubbles} 个<br>"
                summary_text += f"平均每帧: {avg_bubbles:.1f} 个气泡<br>"
            
            # 添加输出信息
            summary_text += f"输出文件夹: {os.path.basename(self.output_folder)}<br>"
            
            summary_text += "</div>"
            
            # 更新信息标签
            self.info_label.setText(summary_text)
            
            self.statusbar.showMessage("界面显示更新完成")
            return True
            
        except Exception as e:
            self.statusbar.showMessage("界面更新出错")
            error_msg = f"更新界面显示时出错: {str(e)}"
            self.add_log(error_msg)
            print(error_msg)
            return False
        
    def finalize_processing(self):
        """完成处理"""
        # 启用导出按钮
        self.export_btn.setEnabled(True)
        if hasattr(self, 'export_action'):
            self.export_action.setEnabled(True)
        
        self.statusbar.showMessage("处理完成")
        
    def handle_processing_error(self, e):
        """处理错误处理"""
        self.statusbar.showMessage("处理出错")
        error_msg = f"处理数据时出错: {str(e)}"
        self.add_log(error_msg)
        QMessageBox.critical(self, "错误", error_msg)
        import traceback
        traceback.print_exc()

    def create_info_panel(self):
        """创建信息面板，包括气泡信息显示和日志显示"""
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(15)
        
        # 创建气泡信息显示组
        bubble_info_group = QGroupBox("气泡信息")
        bubble_info_layout = QVBoxLayout(bubble_info_group)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 创建内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 创建气泡信息标签
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 10px;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
        """)
        content_layout.addWidget(self.info_label)
        
        # 将内容容器添加到滚动区域
        scroll_area.setWidget(content_widget)
        bubble_info_layout.addWidget(scroll_area)
        
        # 创建日志组
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_label = QLabel()
        self.log_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.log_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 10px;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
            }
        """)
        self.log_label.setWordWrap(True)
        log_layout.addWidget(self.log_label)
        
        # 设置组件大小策略
        bubble_info_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        log_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # 添加组到面板
        info_layout.addWidget(bubble_info_group)
        info_layout.addWidget(log_group)
        
        return info_panel
        
    def add_log(self, message):
        """添加日志消息"""
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
            self.log_label.repaint()  # 确保立即更新
        
        # 打印到控制台（调试用）
        print(log_entry)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>BFRM - 气泡流重建模型 v1.0</h2>
        <p>该应用程序用于处理气泡流图像，执行气泡检测和流场重建。</p>
        <p>可以通过加载图像文件夹或直接加载视频文件进行处理。</p>
        <p>© 2024 AI4CT团队</p>
        <p>制作人: 于宝地</p>
        """
        QMessageBox.about(self, "关于气泡流重建模型", about_text)
    def change_playback_speed(self, action):
        """更改播放速度
        
        Args:
            action: 触发的动作对象或速度字符串
        """
        # 从动作获取速度值
        if isinstance(action, str):
            speed_text = action
        else:
            speed_text = action.text()
        
        # 移除"x"后缀并转换为浮点数
        speed_value = float(speed_text.replace('x', ''))
        self.playback_speed = speed_value
        
        # 更新工具栏按钮文本
        if hasattr(self, 'speed_btn'):
            self.speed_btn.setText(f'速度: {speed_text}')
        
        # 更新菜单项选中状态
        if hasattr(self, 'speed_actions'):
            for speed, act in self.speed_actions.items():
                act.setChecked(speed == speed_text)
        
        # 如果是视频模式且正在播放，需要更新定时器
        if self.is_video_mode and self.is_playing:
            self.frame_timer.stop()
            interval = int(1000 / (self.video_fps * self.playback_speed))
            self.frame_timer.start(interval)
        
        # 更新所有动画的播放速度
        for movie in self.movies.values():
            if movie and movie.isValid():
                movie.setSpeed(int(100 * self.playback_speed))
        
        # 更新状态栏显示当前播放速度
        current_status = self.statusbar.currentMessage()
        if "播放速度" in current_status:
            # 如果状态栏已经显示播放速度信息，则更新它
            parts = current_status.split("播放速度:")
            if len(parts) > 1:
                new_status = f"{parts[0]}播放速度: {speed_text}"
                self.statusbar.showMessage(new_status)
        else:
            # 如果没有显示，则添加播放速度信息
            self.statusbar.showMessage(f"{current_status} | 播放速度: {speed_text}")
        
        self.add_log(f"播放速度已更改为: {self.playback_speed}x")
        
        # 如果有信息显示标签，更新播放速度信息
        self.update_display(self.current_frame_index)

    def change_layout(self, action):
        """更改窗口布局
        
        Args:
            action: 触发的动作对象
        """
        layout_text = action.text()
        self.add_log(f"切换布局为: {layout_text}")
        
        if layout_text == '2×2 布局':
            self.set_layout_2x2()
        elif layout_text == '1×4 布局':
            self.set_layout_1x4()
            
    def set_layout_2x2(self):
        """设置2×2网格布局"""
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
            
            # 调整信息面板和主显示区的比例
            if hasattr(self, 'main_horizontal_splitter'):
                self.main_horizontal_splitter.setSizes([800, 200])
        except Exception as e:
            import traceback
            self.add_log(f"设置2×2布局时出错: {str(e)}")
            traceback.print_exc()
        
    def set_layout_1x4(self):
        """设置1×4横向布局"""
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
            
            # 调整信息面板和主显示区的比例
            if hasattr(self, 'main_horizontal_splitter'):
                self.main_horizontal_splitter.setSizes([800, 200])
        except Exception as e:
            import traceback
            self.add_log(f"设置1×4布局时出错: {str(e)}")
            traceback.print_exc()

    def next_frame(self):
        """播放下一帧"""
        # 如果没有帧，直接返回
        if not self.frames:
            return
            
        # 计算下一帧索引
        next_index = self.current_frame_index + 1
        
        # 检查是否到达已加载的视频末尾
        if next_index >= len(self.frames):
            if self.loop_checkbox.isChecked():
                # 循环播放，回到第一帧
                next_index = 0
                # 清空气泡轨迹
                self.trajectories = {}
                self.add_log("循环播放：清空气泡轨迹")
            else:
                # 暂停播放，但保持在最后一帧
                self.pause_playback()
                # 确保显示最后一帧
                if len(self.frames) > 0:
                    next_index = len(self.frames) - 1
                    # 更新当前帧索引
                    self.current_frame_index = next_index
                    # 更新滑动条位置（不触发valueChanged信号）
                    self.frame_slider.blockSignals(True)
                    self.frame_slider.setValue(next_index)
                    self.frame_slider.blockSignals(False)
                    # 更新显示
                    self.update_display(next_index)
                return
        
        # 更新当前帧索引
        self.current_frame_index = next_index
        
        # 更新滑动条位置（不触发valueChanged信号）
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(next_index)
        self.frame_slider.blockSignals(False)
        
        # 更新显示
        self.update_display(next_index)

    def update_display(self, frame_index):
        """更新显示"""
        if not hasattr(self, 'frames') or not self.frames or frame_index >= len(self.frames):
            return

        # 更新帧号标签
        if hasattr(self, 'frame_number_label'):
            total_frames = len(self.frames)
            self.frame_number_label.setText(f"{frame_index + 1}/{total_frames}")

        # 获取当前帧
        frame = self.frames[frame_index]
        
        # 使用YOLO处理当前帧
        processed_frame, frame_info = self.process_frame_with_yolo(frame, frame_index)
        
        # 更新气泡信息显示
        self.update_bubble_info_display(frame_info)
        
        # 更新原始视频显示
        if hasattr(self, 'content_labels') and self.content_labels.get('flow'):
            self._update_label_with_image(self.content_labels['flow'], frame)

        # 更新检测结果显示
        if hasattr(self, 'content_labels') and self.content_labels.get('detection'):
            self._update_label_with_image(self.content_labels['detection'], processed_frame)

        # 更新3D重建图像（如果有）
        if hasattr(self, 'content_labels') and self.content_labels.get('3d'):
            # TODO: 实现3D重建显示
            pass

        # 更新简化流场图像（如果有）
        if hasattr(self, 'content_labels') and self.content_labels.get('empty'):
            # TODO: 实现简化流场显示
            pass

        # 更新状态栏，显示当前帧信息
        self.statusbar.showMessage(f"当前帧: {frame_index + 1}/{len(self.frames)}, 播放速度: {self.playback_speed}x")
        
    def _count_bubbles(self, frame_index):
        """计算当前帧检测到的气泡数量"""
        if frame_index in self.frame_info_dict and 'bubbles' in self.frame_info_dict[frame_index]:
            return len(self.frame_info_dict[frame_index]['bubbles'])
        return 0
        
    def _average_bubbles_per_frame(self):
        """计算平均每帧的气泡数量"""
        total_bubbles = 0
        frames_with_data = 0
        
        for frame_idx, info in self.frame_info_dict.items():
            if 'bubbles' in info:
                total_bubbles += len(info['bubbles'])
                frames_with_data += 1
                
        if frames_with_data > 0:
            return round(total_bubbles / frames_with_data, 1)
        return 0.0

    def _update_label_with_image(self, label, frame):
        """更新标签显示图像
        
        Args:
            label: QLabel对象，用于显示图像
            frame: OpenCV格式的图像（BGR）
        """
        if frame is None:
            return
            
        # 将OpenCV的BGR格式转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 创建QPixmap并设置到标签
        pixmap = QPixmap.fromImage(q_img)
        
        # 缩放图像以适应标签大小，保持纵横比
        label_size = label.size()
        scaled_pixmap = pixmap.scaled(
            label_size.width() - 10, 
            label_size.height() - 10,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def toggle_play_pause(self):
        """切换播放/暂停状态"""
        if not self.frames or len(self.frames) == 0:
            return
            
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """开始播放视频"""
        if not self.frames or len(self.frames) == 0:
            return
            
        self.is_playing = True
        
        # 计算帧间隔时间（毫秒）
        interval = int(1000 / (self.video_fps * self.playback_speed))
        self.frame_timer.start(interval)
        
        self.add_log("开始播放视频")
    
    def pause_playback(self):
        """暂停播放视频"""
        if not self.is_playing:
            return
            
        self.frame_timer.stop()
        self.is_playing = False
        
        self.add_log("暂停视频播放")
    
    def stop_playback(self):
        """停止播放视频"""
        if not self.frames:
            return
            
        # 停止播放
        self.frame_timer.stop()
        self.is_playing = False
        
        # 重置到第一帧
        self.current_frame_index = 0
        self.frame_slider.setValue(0)
        self.update_display(0)
        
        self.add_log("停止视频播放")
    
    def replay_animations(self):
        """重新播放动画"""
        if not self.frames or len(self.frames) == 0:
            return
            
        # 停止当前播放
        self.stop_playback()
        
        # 重置气泡轨迹
        self.trajectories = {}
        
        # 重置到第一帧
        self.current_frame_index = 0
        self.frame_slider.setValue(0)
        
        # 更新显示
        self.update_display(0)
        
        # 开始播放
        self.start_playback()
        
        self.add_log("重新开始播放视频")
    
    def frame_slider_changed(self, value):
        """滑动条值改变事件处理
        
        Args:
            value: 新的滑动条值
        """
        if not self.frames or len(self.frames) == 0:
            return
            
        # 暂停播放（如果正在播放）
        if self.is_playing:
            self.pause_playback()
        
        # 更新当前帧索引
        self.current_frame_index = value
        
        # 更新显示
        self.update_display(value)

    def toggle_loop(self):
        """切换循环播放设置"""
        is_loop = self.loop_checkbox.isChecked()
        self.add_log(f"循环播放: {'开启' if is_loop else '关闭'}")
        
    def export_video(self):
        """导出处理后的视频"""
        # 检查是否有可用的帧
        if not self.frames or len(self.frames) == 0:
            self.statusbar.showMessage("没有可用的视频帧")
            QMessageBox.warning(self, "导出错误", "没有可用的视频帧可导出！")
            return
            
        # 选择保存位置
        suggested_name = "processed_video.mp4" if self.is_video_mode else "animated_sequence.mp4"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出视频",
            os.path.join(self.output_folder, suggested_name),
            "视频文件 (*.mp4)"
        )
        
        if not file_path:
            return
            
        try:
            self.statusbar.showMessage("正在导出视频...")
            self.progress_bar.setValue(0)
            self.add_log(f"开始导出视频到: {file_path}")
            
            # 获取第一帧尺寸
            height, width = self.frames[0].shape[:2]
            
            # 创建VideoWriter对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, self.video_fps, (width, height))
            
            # 写入每一帧
            total_frames = len(self.frames)
            for i, frame in enumerate(self.frames):
                # 计算进度
                progress = int((i + 1) / total_frames * 100)
                self.progress_bar.setValue(progress)
                
                # 写入帧
                out.write(frame)
                
                # 每10帧更新一次状态
                if (i + 1) % 10 == 0 or i == total_frames - 1:
                    self.statusbar.showMessage(f"正在导出视频... {progress}%")
                    QApplication.processEvents()
            
            # 释放资源
            out.release()
            
            self.statusbar.showMessage(f"视频导出完成: {file_path}")
            self.progress_bar.setValue(100)
            self.add_log(f"视频导出完成: {file_path}")
            
            # 询问是否打开视频
            reply = QMessageBox.question(
                self,
                "导出完成",
                f"视频已成功导出到: {file_path}\n\n是否打开视频？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                import webbrowser
                webbrowser.open(file_path)
            
        except Exception as e:
            self.statusbar.showMessage("导出视频失败")
            error_msg = f"导出视频时出错: {str(e)}"
            self.add_log(error_msg)
            QMessageBox.critical(self, "导出错误", error_msg)
            import traceback
            traceback.print_exc()

    def keyPressEvent(self, event):
        """处理键盘按键事件"""
        # 获取按键
        key = event.key()
        
        # 检查键盘按键并执行相应操作
        if key == 32:  # Space
            # 空格键控制播放/暂停
            if self.is_playing:
                self.pause_playback()
            else:
                self.start_playback()
        elif key == 16777234:  # Left
            # 左箭头键，后退一帧
            if self.current_frame_index > 0:
                self.current_frame_index -= 1
                self.update_frame_slider(self.current_frame_index)
                self.update_display(self.current_frame_index)
        elif key == 16777236:  # Right
            # 右箭头键，前进一帧
            if self.current_frame_index < len(self.frames) - 1:
                self.current_frame_index += 1
                self.update_frame_slider(self.current_frame_index)
                self.update_display(self.current_frame_index)
        elif key == 43 or key == 61:  # Plus or Equal
            # +键，增加播放速度
            speeds = ['0.25x', '0.5x', '1.0x', '1.5x', '2.0x', '4.0x']
            current_index = 0
            # 找到当前速度的索引
            for i, speed in enumerate(speeds):
                if float(speed.replace('x', '')) == self.playback_speed:
                    current_index = i
                    break
            # 如果不是最大速度，则提高速度
            if current_index < len(speeds) - 1:
                self.change_playback_speed(speeds[current_index + 1])
        elif key == 45:  # Minus
            # -键，减少播放速度
            speeds = ['0.25x', '0.5x', '1.0x', '1.5x', '2.0x', '4.0x']
            current_index = 0
            # 找到当前速度的索引
            for i, speed in enumerate(speeds):
                if float(speed.replace('x', '')) == self.playback_speed:
                    current_index = i
                    break
            # 如果不是最小速度，则降低速度
            if current_index > 0:
                self.change_playback_speed(speeds[current_index - 1])
        else:
            # 其他按键，交给父类处理
            super().keyPressEvent(event)

    def init_yolo_model(self):
        """初始化YOLO模型"""
        try:
            from ultralytics import YOLO
            import os
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            
            self.add_log("正在初始化YOLO模型...")
            self.model = YOLO(r'C:\codebase\yolo\model\yolo11n-obb.pt')
            self.add_log("YOLO模型初始化完成")
            return True
        except Exception as e:
            self.add_log(f"YOLO模型初始化失败: {str(e)}")
            return False
            
    def process_frame_with_yolo(self, frame, frame_index):
        """使用YOLO模型处理单帧图像
        
        Args:
            frame: 输入图像帧
            frame_index: 帧索引
            
        Returns:
            processed_frame: 处理后的图像帧
            frame_info: 当前帧的气泡信息
        """
        import cv2
        import numpy as np
        
        if self.model is None:
            if not self.init_yolo_model():
                return frame, {}
        
        # 复制输入帧
        processed_frame = frame.copy()
        frame_info = {}
        
        try:
            # 使用YOLO模型进行检测，禁用终端输出
            results = self.model.track(
                processed_frame,
                persist=True,
                show_boxes=True,
                tracker='botsort.yaml',
                verbose=False
            )
            
            if results and len(results) > 0:
                r = results[0]  # 获取第一个结果
                
                # 设置绘制参数
                lw = max(round(sum(processed_frame.shape) / 2 * 0.002), 2)
                tf = max(lw - 1, 1)
                sf = lw / 3
                
                # 处理每个检测到的气泡
                for i in range(len(r.obb)):
                    box = r.obb.xyxyxyxy[i].reshape(-1, 4, 2).squeeze()
                    rbox = r.obb.xywhr[i]
                    center = (rbox[0], rbox[1])
                    
                    # 获取气泡ID
                    bubble_id = int(r.obb.id[i]) if hasattr(r.obb, 'id') else i
                    
                    # 更新气泡轨迹
                    if bubble_id not in self.trajectories:
                        self.trajectories[bubble_id] = []
                    self.trajectories[bubble_id].append(center)
                    if len(self.trajectories[bubble_id]) > 100:
                        self.trajectories[bubble_id].pop(0)
                    
                    # 绘制轨迹
                    color = (0, 0, 192) if r.obb.cls[i] else (53, 130, 84)
                    for j in range(1, len(self.trajectories[bubble_id])):
                        start_point = tuple(map(int, self.trajectories[bubble_id][j-1]))
                        end_point = tuple(map(int, self.trajectories[bubble_id][j]))
                        cv2.line(processed_frame, start_point, end_point, color, 2)
                    
                    # 绘制检测框
                    cv2.polylines(processed_frame, [np.asarray(box, dtype=int)], True, color, lw)
                    
                    # 计算气泡信息
                    w, h = rbox[2], rbox[3]
                    angle = rbox[4] * 180 / np.pi
                    
                    # 计算速度
                    speed = 0
                    if len(self.trajectories[bubble_id]) >= 2:
                        speed = np.linalg.norm(
                            np.array(self.trajectories[bubble_id][-1]) - 
                            np.array(self.trajectories[bubble_id][-2])
                        ) * 0.080128 * 0.001 / 0.00001
                    
                    # 计算体积
                    volume = (w * w * h * 0.080128 ** 3)
                    
                    # 存储气泡信息
                    frame_info[bubble_id] = {
                        'id': bubble_id,
                        'x': center[0],
                        'y': center[1],
                        'width': w,
                        'height': h,
                        'angle': angle,
                        'speed': speed,
                        'volume': volume
                    }
            
            return processed_frame, frame_info
            
        except Exception as e:
            self.add_log(f"处理帧 {frame_index} 时出错: {str(e)}")
            return frame, {}

    def update_bubble_info_display(self, frame_info):
        """更新气泡信息显示
        
        Args:
            frame_info: 当前帧的气泡信息字典
        """
        if not frame_info:
            if hasattr(self, 'info_label'):
                self.info_label.setText("<div style='text-align:center; padding:20px;'>暂无气泡信息</div>")
            return
            
        # 创建HTML表格
        table_html = """
        <style>
            .container {
                font-family: Arial, sans-serif;
                font-size: 13px;
                color: #333;
            }
            .header {
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
                background-color: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 5px;
            }
            th {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 4px;
                position: sticky;
                top: 0;
            }
            td {
                padding: 6px 4px;
                border-bottom: 1px solid #ddd;
            }
            tr:nth-child(even) {
                background-color: #f8f8f8;
            }
            tr:hover {
                background-color: #f0f0f0;
            }
        </style>
        <div class='container'>
            <div class='header'>
                当前检测到 {len(frame_info)} 个气泡
            </div>
            <table>
                <tr>
                    <th>气泡ID</th>
                    <th>位置(x,y)</th>
                    <th>尺寸(w×h)</th>
                    <th>角度(°)</th>
                    <th>速度(m/s)</th>
                    <th>体积(mm³)</th>
                </tr>
        """
        
        # 按气泡ID排序
        sorted_bubbles = sorted(frame_info.items(), key=lambda x: x[0])
        
        for bubble_id, info in sorted_bubbles:
            table_html += f"""
                <tr>
                    <td>{info['id']}</td>
                    <td>({info['x']:.1f}, {info['y']:.1f})</td>
                    <td>{info['width']:.1f}×{info['height']:.1f}</td>
                    <td>{info['angle']:.1f}</td>
                    <td>{info['speed']:.2f}</td>
                    <td>{info['volume']:.2f}</td>
                </tr>
            """
        
        table_html += """
            </table>
        </div>
        """
        
        # 更新信息标签
        if hasattr(self, 'info_label'):
            self.info_label.setText(table_html)
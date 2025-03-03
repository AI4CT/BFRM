import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO  # 导入YOLO库
from collections import deque

class YoloTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLO 测试')
        self.setGeometry(100, 100, 1200, 600)

        # 创建主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 创建水平布局
        self.horizontal_layout = QHBoxLayout()

        # 创建检测结果显示标签
        self.result_label = QLabel('检测结果区域')
        self.result_label.setFont(QFont('Arial', 16))  # 设置字体大小
        self.result_label.setMinimumSize(400, 100)  # 设置最小尺寸
        self.horizontal_layout.addWidget(self.result_label)

        # 创建原始视频显示标签
        self.original_video_label = QLabel('原始视频播放区域')
        self.original_video_label.setMinimumSize(800, 400)  # 设置最小尺寸
        self.horizontal_layout.addWidget(self.original_video_label)

        # 创建视频显示标签
        self.video_label = QLabel('视频播放区域')
        self.video_label.setMinimumSize(800, 400)  # 设置最小尺寸
        self.horizontal_layout.addWidget(self.video_label)

        # 将水平布局添加到主布局中
        self.layout.addLayout(self.horizontal_layout)

        # 创建表格显示检测信息
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(8)  # 8列：类别、坐标、尺寸、体积、X速度、Y速度、合速度、角度
        self.table_widget.setHorizontalHeaderLabels(['类别', '坐标 (像素)', '尺寸 (mm)', '体积 (mm³)', 'X速度 (m/s)', 'Y速度 (m/s)', '合速度 (m/s)', '角度 (°)'])
        self.layout.addWidget(self.table_widget)

        # 创建按钮
        self.load_button = QPushButton('加载视频')
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        # 初始化 YOLO 模型
        self.model = YOLO('C:\\codebase\\yolo\\model\\yolo11m-obb.pt')  # 加载模型
        print("YOLO模型加载成功")  # 调试信息

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.frame_time = 0.00001  # 每帧时间
        self.pixel_to_mm = 0.080128  # 每像素的长度

        # 允许窗口调节尺寸
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 存储前一帧的信息
        self.prev_frame_bubbles = {}  # 存储前一帧中每个气泡的信息 {id: (x, y, w, h, angle)}
        self.current_frame_bubbles = {}  # 存储当前帧中每个气泡的信息
        self.bubble_ids = {}  # 用于跟踪气泡ID {(x, y): id}
        self.next_bubble_id = 0  # 下一个可用的气泡ID

        # 在类的初始化方法中添加以下内容
        self.trackers = []  # 存储追踪器
        self.track_id = 0  # 追踪ID
        self.tracked_objects = {}  # 存储被追踪的对象
        self.max_distance = 50  # 最大距离阈值
        self.tracking_history = {}  # 存储每个对象的历史位置

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '', '视频文件 (*.mp4 *.avi *.mov *.mkv)')
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.start(50)  # 每50毫秒更新一次帧
            print(f"加载视频: {video_path}")  # 调试信息

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # 显示原始视频帧
                self.display_original_video(frame)
                # 显示视频帧
                self.display_video(frame)
                # 进行目标检测
                self.detect_objects(frame)
            else:
                self.timer.stop()
                self.cap.release()
                print("视频读取完毕")  # 调试信息

    def display_original_video(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.original_video_label.setPixmap(pixmap.scaled(self.original_video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def display_video(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # 自动调整窗口大小
        self.resize(w, h + 100)  # 100是额外的空间用于其他控件

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)  # 进行目标检测
        self.display_results(results, frame)  # 传递frame

    def display_results(self, results, frame):
        # 清空表格
        self.table_widget.setRowCount(0)

        # 获取检测结果
        result = results[0].obb
        cls = result.cls
        conf = result.conf
        xywhr = result.xywhr.cpu()  # 将张量移动到CPU
        xyxyxyxys = result.xyxyxyxy.cpu()  # 将张量移动到CPU
        orig_shape = result.orig_shape

        # 更新当前帧气泡信息
        self.current_frame_bubbles.clear()

        # 绘制检测框
        for i in range(len(cls)):
            points = xywhr[i].numpy()  # 转换为numpy数组
            xyxyxyxy = xyxyxyxys[i].numpy().astype(int)  # 转换为整数
            color = (53, 130, 84) if cls[i] == 0 else (0, 0, 192)  # BGR颜色
            cv2.polylines(frame, [np.asarray(xyxyxyxy, dtype=int)], isClosed=True, color=color, thickness=2)

            # 获取气泡中心坐标
            center_x, center_y = points[0], points[1]
            
            # 为新气泡分配ID或找到已存在气泡的ID
            min_dist = float('inf')
            matched_id = None
            
            # 在前一帧中寻找最近的气泡
            for prev_id, prev_info in self.prev_frame_bubbles.items():
                prev_x, prev_y = prev_info[0], prev_info[1]
                dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                if dist < min_dist and dist < self.max_distance:  # 使用阈值避免错误匹配
                    min_dist = dist
                    matched_id = prev_id

            # 如果没有找到匹配的气泡，分配新ID
            if matched_id is None:
                matched_id = self.next_bubble_id
                self.next_bubble_id += 1

            # 存储当前气泡信息
            self.current_frame_bubbles[matched_id] = (center_x, center_y, points[2], points[3], points[4])

            # 显示检测信息
            bubble_type = "single" if cls[i] == 0 else "overlap"
            self.result_label.setText(f'类别: {bubble_type}, 置信度: {conf[i]:.2f}, 原始尺寸: {orig_shape}')

            # 更新表格
            self.table_widget.insertRow(i)
            self.table_widget.setItem(i, 0, QTableWidgetItem(bubble_type))
            
            # 显示坐标（保留两位小数）
            self.table_widget.setItem(i, 1, QTableWidgetItem(f'({center_x:.2f}, {center_y:.2f})'))

            # 从xywhr获取宽度、高度，并转换为毫米
            width_mm = float(points[2]) * self.pixel_to_mm  # 宽度转换为毫米
            height_mm = float(points[3]) * self.pixel_to_mm  # 高度转换为毫米
            # 确保宽度大于高度
            if width_mm < height_mm:
                width_mm, height_mm = height_mm, width_mm
            self.table_widget.setItem(i, 2, QTableWidgetItem(f'宽: {width_mm:.2f}, 高: {height_mm:.2f}'))

            # 计算体积（假设气泡是三维椭球）
            min_axis = min(width_mm, height_mm)  # 取最小轴作为垂直于图像平面的轴长
            volume_mm3 = (4/3) * np.pi * (width_mm/2) * (height_mm/2) * (min_axis/2)  # 椭球体积公式
            self.table_widget.setItem(i, 3, QTableWidgetItem(f'{volume_mm3:.2f}'))

            # 计算速度（仅当能找到匹配的前一帧气泡时）
            if matched_id in self.prev_frame_bubbles:
                prev_x, prev_y = self.prev_frame_bubbles[matched_id][0:2]
                # 计算位移（像素）
                dx = center_x - prev_x
                dy = center_y - prev_y
                # 转换为米（先转毫米再转米）
                dx_m = dx * self.pixel_to_mm / 1000.0
                dy_m = dy * self.pixel_to_mm / 1000.0
                # 计算速度（米/秒）
                speed_x = dx_m / self.frame_time
                speed_y = dy_m / self.frame_time
                speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)
            else:
                speed_x = speed_y = speed_magnitude = 0.0

            # 显示速度
            self.table_widget.setItem(i, 4, QTableWidgetItem(f'{speed_x:.2f}'))
            self.table_widget.setItem(i, 5, QTableWidgetItem(f'{speed_y:.2f}'))
            self.table_widget.setItem(i, 6, QTableWidgetItem(f'{speed_magnitude:.2f}'))

            # 显示角度（转换为角度制）
            angle_deg = np.degrees(points[4])  # 将弧度转换为角度
            self.table_widget.setItem(i, 7, QTableWidgetItem(f'{angle_deg:.2f}°'))

        # 更新前一帧信息
        self.prev_frame_bubbles = self.current_frame_bubbles.copy()

        # 显示带有检测框的帧
        self.display_video(frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YoloTestApp()
    window.show()
    print("应用程序启动")  # 调试信息
    sys.exit(app.exec()) 
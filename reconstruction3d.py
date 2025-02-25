import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab  # 使用mayavi替代matplotlib实现更好的3D交互
from vispy import scene, app
from vispy.geometry import create_sphere, create_box
from vispy.visuals.transforms import MatrixTransform, STTransform
from PyQt6.QtWidgets import (QWidget, QColorDialog, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QGridLayout, QLabel)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

def calculate_ellipsoid_params(csv_data):
    """根据CSV数据计算椭球体参数"""
    # 从2D椭圆参数推算3D椭球体参数
    major_axis = csv_data['major_axis']
    minor_axis = csv_data['minor_axis']
    angle = csv_data['angle']
    # ... 计算逻辑 ...
    return a, b, c, theta, phi

class Interactive3DCanvas(QWidget):
    """交互式3D画布"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除布局边距
        
        # 创建画布
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), bgcolor='white')
        self.canvas.create_native()
        
        # 创建容器widget来放置画布和比例尺
        container = QWidget()
        container_layout = QGridLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(container_layout)
        
        # 添加画布到容器
        container_layout.addWidget(self.canvas.native, 0, 0, 1, 1)
        
        # 创建比例尺标签
        self.scale_widget = QWidget()
        self.scale_widget.setFixedHeight(30)  # 设置固定高度
        scale_layout = QHBoxLayout()
        scale_layout.setContentsMargins(0, 0, 0, 0)
        self.scale_widget.setLayout(scale_layout)
        
        # 创建比例尺线和文字
        self.scale_label = QLabel()
        self.scale_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.scale_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: black;
                font-size: 12px;
            }
        """)
        
        # 使用HTML来绘制比例尺
        scale_html = """
        <div style='text-align: center;'>
            <div style='border-top: 2px solid black; width: 100%;'></div>
            <div>10mm</div>
        </div>
        """
        self.scale_label.setText(scale_html)
        self.scale_label.setVisible(False)  # 初始状态隐藏
        
        # 添加弹簧和比例尺标签到比例尺布局
        scale_layout.addStretch(7)  # 左侧占70%
        scale_layout.addWidget(self.scale_label, stretch=3)  # 比例尺占30%
        
        # 将比例尺widget添加到容器
        container_layout.addWidget(self.scale_widget, 0, 0, 1, 1, Qt.AlignmentFlag.AlignBottom)
        
        # 添加容器到主布局
        layout.addWidget(container)
        
        # 创建3D视图
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=0, elevation=30, azimuth=45)
        
        # 设置基本参数
        self.flow_width = 137
        self.flow_depth = self.flow_width
        self.current_flow_height = 0  # 初始化当前流场高度
        
        # 设置相机中心位置到流场中心
        self.view.camera.center = (0, 0, 0)  # 初始设置为原点
        self.view.camera.scale_factor = self.flow_width  # 初始缩放因子
        
        # 设置颜色
        self.bubble_color = QColor(0, 100, 200, 128)   # 深蓝色，半透明
        
        # 添加首次更新标志
        self.is_first_update = True
        
        # 创建边界线（所有顶点相连）
        vertices = np.array([
            # 前面
            [0, 0, 0], [1, 0, 0],  # 底边
            [1, 0, 0], [1, 1, 0],  # 右边
            [1, 1, 0], [0, 1, 0],  # 顶边
            [0, 1, 0], [0, 0, 0],  # 左边

            # 后面
            [0, 0, 1], [1, 0, 1],  # 底边
            [1, 0, 1], [1, 1, 1],  # 右边
            [1, 1, 1], [0, 1, 1],  # 顶边
            [0, 1, 1], [0, 0, 1],  # 左边

            # 连接线
            [0, 0, 0], [0, 0, 1],  # 左下
            [1, 0, 0], [1, 0, 1],  # 右下
            [1, 1, 0], [1, 1, 1],  # 右上
            [0, 1, 0], [0, 1, 1]   # 左上
        ])
        
        self.box_lines = scene.visuals.Line(
            pos=vertices,
            color='black',
            parent=self.view.scene,
            connect='segments',
            width=2
        )
        self.box_lines.visible = False  # 初始状态隐藏
        
        # 创建坐标轴
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        
        # 添加坐标轴标签
        self.x_label = scene.visuals.Text(
            pos=[1.2, 0, 0], text='X', color='red', 
            parent=self.view.scene, font_size=12
        )
        self.y_label = scene.visuals.Text(
            pos=[0, 1.2, 0], text='Y', color='green',
            parent=self.view.scene, font_size=12
        )
        self.z_label = scene.visuals.Text(
            pos=[0, 0, 1.2], text='Z', color='blue',
            parent=self.view.scene, font_size=12
        )
        
        # 创建气泡（使用球体近似）
        rows, cols = 20, 20
        phi = np.linspace(0, 2*np.pi, cols)
        theta = np.linspace(0, np.pi, rows)
        phi, theta = np.meshgrid(phi, theta)
        
        x = 0.5 * np.sin(theta) * np.cos(phi)
        y = 0.5 * np.sin(theta) * np.sin(phi)
        z = 0.5 * np.cos(theta)
        
        vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        
        # 创建面索引
        faces = []
        for i in range(rows-1):
            for j in range(cols-1):
                faces.append([i*cols + j, i*cols + j+1, (i+1)*cols + j])
                faces.append([i*cols + j+1, (i+1)*cols + j+1, (i+1)*cols + j])
        faces = np.array(faces, dtype=np.uint32)
        
        self.bubble = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=self.bubble_color.getRgbF(),
            shading='smooth',
            parent=self.view.scene
        )
        self.bubble.visible = False  # 初始状态隐藏
        
        # 创建重置视图按钮
        self.reset_view_btn = QPushButton('重置视图', self)
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.reset_view_btn)
        
        # 设置初始变换
        self.bubble_transform = STTransform()
        self.bubble.transform = self.bubble_transform
        self.box_lines_transform = STTransform()
        self.box_lines.transform = self.box_lines_transform
        
        # 设置视图范围
        self.view.camera.set_range()
        
        # 添加鼠标事件处理
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        
        # 启用鼠标滚轮事件
        self.canvas.native.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        
        self.is_dragging = False
        self.last_pos = None
        
    def resizeEvent(self, event):
        """处理窗口大小变化"""
        super().resizeEvent(event)
        
    def reset_view(self):
        """重置相机视角和视野"""
        # 设置正视图（与二维视图一致）
        self.view.camera.elevation = 0
        self.view.camera.azimuth = 0
        self.view.camera.fov = 0
        
        # 计算流场尺寸
        flow_width = 137
        flow_height = 314
        flow_depth = flow_width
        
        # 设置相机中心到流场中心
        center_x = 0
        center_y = 0
        center_z = flow_height / 2
        self.view.camera.center = (center_x, center_y, center_z)
        
        # 计算合适的缩放因子，使流场填充显示区域
        canvas_size = self.canvas.size
        aspect_ratio = canvas_size[0] / canvas_size[1]
        
        # 根据画布宽高比和流场尺寸计算合适的缩放因子
        if aspect_ratio > flow_width / flow_height:
            # 画布较宽，以高度为基准
            scale_factor = flow_height * 1.1  # 留出10%边距
        else:
            # 画布较高，以宽度为基准
            scale_factor = flow_width * 1.1 / aspect_ratio
            
        self.view.camera.scale_factor = scale_factor
        
        # 更新显示
        self.canvas.update()
        
    def change_bubble_color(self):
        """更改气泡颜色"""
        color = QColorDialog.getColor(self.bubble_color, self, "选择气泡颜色",
                                    QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self.bubble_color = color
            self.bubble.color = color.getRgbF()
            self.canvas.update()
        
    def on_mouse_press(self, event):
        """处理鼠标按下事件"""
        if event.button == 2:  # 鼠标中键
            self.is_dragging = True
            self.last_pos = event.pos
            
    def on_mouse_release(self, event):
        """处理鼠标释放事件"""
        if event.button == 2:  # 鼠标中键
            self.is_dragging = False
            
    def on_mouse_move(self, event):
        """处理鼠标移动事件"""
        if self.is_dragging and self.last_pos is not None:
            # 计算移动距离
            dx = event.pos[0] - self.last_pos[0]
            dy = event.pos[1] - self.last_pos[1]
            
            # 更新相机位置
            self.view.camera.azimuth += dx * 0.5
            self.view.camera.elevation += dy * 0.5
            
            # 更新上一次位置
            self.last_pos = event.pos
            
            # 更新显示
            self.canvas.update()
        
    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件以控制视野缩放"""
        try:
            # 获取滚轮的增量并简化处理
            if hasattr(event, 'native') and hasattr(event.native, 'angleDelta'):
                # 使用原生事件
                delta = event.native.angleDelta().y()
            else:
                # 使用vispy事件
                delta = event.delta[1] * 120.0
                
            # 简化缩放逻辑
            if delta > 0:  # 向上滚动，放大
                scale_factor = 0.9  # 缩小scale_factor意味着视图放大
            else:  # 向下滚动，缩小
                scale_factor = 1.1  # 增大scale_factor意味着视图缩小
            
            # 获取当前缩放因子
            current_scale = self.view.camera.scale_factor
            
            # 计算新的缩放因子，并限制在合理范围内
            new_scale = current_scale * scale_factor
            min_scale = self.flow_width * 0.1  # 最小缩放
            max_scale = self.flow_width * 10   # 最大缩放
            
            # 应用缩放限制
            new_scale = max(min_scale, min(new_scale, max_scale))
            
            # 设置新的缩放因子
            self.view.camera.scale_factor = new_scale
            
            # 强制更新视图
            self.view.camera.view_changed()
            self.canvas.update()
            
            # 阻止事件继续传播
            event.handled = True
            
        except Exception as e:
            print(f"鼠标滚轮事件处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def update_bubble(self, data_row):
        """更新气泡显示"""
        if data_row is None:
            return
            
        try:
            # 获取气泡参数
            x = float(data_row['x0'])
            y = float(data_row['y0'])
            w = float(data_row['w_r'])
            h = float(data_row['h_r'])
            theta = float(data_row['theta'])
            
            # 更新流场高度（取当前y坐标的最大值）
            current_y = y + 64  # 添加偏移
            self.current_flow_height = max(self.current_flow_height, current_y + h)  # 考虑气泡高度
            
            # 只在首次更新时重置视角和视野大小
            if self.is_first_update:
                self.reset_view()
                self.is_first_update = False
            
            # 计算z轴长度（等于宽度）
            z_length = w
            
            # 显示气泡和轮廓线
            self.bubble.visible = True
            self.box_lines.visible = True
            self.scale_label.setVisible(True)  # 显示比例尺
            
            # 更新流场区域大小
            flow_width = self.flow_width
            flow_height = self.current_flow_height
            flow_depth = self.flow_depth
            
            # 计算流场中心位置（x和y方向居中，z方向从0开始）
            center_x = flow_width / 2
            center_y = flow_depth / 2
            center_z = 0  # 从z=0开始
            
            # 更新气泡变换（相对于流场中心，y轴映射到z轴）
            self.bubble_transform.translate = [
                center_x + (x - center_x),  # 保持x轴相对位置
                0,                          # y轴置0
                flow_height - current_y     # 翻转z坐标，从顶部开始向下运动
            ]
            self.bubble_transform.scale = [w/2, z_length/2, h/2]  # 交换y和z轴的缩放
            
            # 创建旋转矩阵（绕y轴旋转，theta已经是弧度制）
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            self.bubble_transform.rotate = rotation_matrix
            
            # 更新流场变换（将流场移动到正确位置，使底部对齐z=0）
            self.box_lines_transform.translate = [-center_x, -center_y, 0]  # z轴从0开始
            self.box_lines_transform.scale = [flow_width, flow_depth, flow_height]  # 交换高度和深度
            
            # 更新坐标轴长度（1.5倍流场尺寸）
            axis_scale = 1.5
            self.axis.transform = STTransform(scale=[
                flow_width * axis_scale,
                flow_depth * axis_scale,    # 深度映射到y轴
                flow_height * axis_scale    # 高度映射到z轴
            ])
            
            # 更新坐标轴标签位置
            self.x_label.pos = [flow_width * axis_scale, 0, 0]
            self.y_label.pos = [0, flow_depth * axis_scale, 0]
            self.z_label.pos = [0, 0, flow_height * axis_scale]
            
            # 更新相机中心位置（但不改变缩放因子）
            self.view.camera.center = (0, 0, flow_height / 2)
            
            # 更新显示
            self.canvas.update()
            
        except Exception as e:
            print(f"更新气泡显示时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def clear(self):
        """清除显示"""
        # 重置变换
        self.bubble_transform.translate = [0, 0, 0]
        self.bubble_transform.scale = [1, 1, 1]
        self.bubble_transform.rotate = 0
        self.box_lines_transform.translate = [0, 0, 0]
        self.box_lines_transform.scale = [1, 1, 1]
        self.axis.transform = STTransform(scale=[1, 1, 1])
        
        # 重置流场高度
        self.current_flow_height = 0
        
        # 隐藏比例尺
        self.scale_label.setVisible(False)
        
        # 重置首次更新标志
        self.is_first_update = True
        # 更新显示
        self.canvas.update()

def reconstruct_3d(data_folder, output_folder):
    """
    根据气泡的椭圆参数进行三维重建。
    返回Interactive3DCanvas对象。
    """
    return Interactive3DCanvas() 
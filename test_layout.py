import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QMenu, 
                             QFrame, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QAction, QActionGroup

class TestLayoutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("布局测试程序")
        self.resize(1200, 800)
        self.init_ui()
    
    def init_ui(self):
        # 创建中心部件和主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建主显示区域
        self.create_main_display_area()
        
        # 显示窗口
        self.show()
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        
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
        
        # 创建右侧信息面板
        info_panel = QFrame()
        info_panel.setFrameShape(QFrame.Shape.Box)
        info_panel.setMinimumWidth(200)
        
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
        
        # 添加标题标签
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # 创建内容标签（可以加入测试图像）
        content_label = QLabel("内容区域")
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_label.setStyleSheet("background-color: #f0f0f0;")
        content_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(content_label)
        
        # 设置最小大小
        frame.setMinimumSize(200, 200)
        
        return frame
    
    def recreate_display_labels(self):
        self.labelFlow = self.create_display_label("区域一")
        self.labelDetection = self.create_display_label("区域二")
        self.label3D = self.create_display_label("区域三")
        self.labelEmpty = self.create_display_label("区域四")
    
    def change_layout(self, action):
        layout_text = action.text()
        print(f"切换布局为: {layout_text}")
        
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

def main():
    app = QApplication(sys.argv)
    window = TestLayoutWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from gui import MainWindow

def main():
    # 先创建QApplication实例
    app = QApplication(sys.argv)
    
    # 然后设置默认字体
    font = QFont('Arial', 10)  # 可以根据需要调整字体和大小
    app.setFont(font)
    
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
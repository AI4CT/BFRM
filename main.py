import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from gui import MainWindow
# 这个版本很好，2025年2月25日晚10点
def main():
    # 设置默认字体
    font = QFont('Arial', 10)  # 可以根据需要调整字体和大小
    QApplication.setFont(font)
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
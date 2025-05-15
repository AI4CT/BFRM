#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BFRM - Bubble Flow Reconstruction and Measurement (Realtime Version)
Created by: AI4CT Team
Author: BaodI Yu (yubaodi20@ipe.ac.cn)
GitHub: https://github.com/AI4CT/BFRM
Copyright (c) 2024 AI4CT Team. All rights reserved.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QIcon
from realtime_gui import RealtimeMainWindow
from config import Config
import logging
from datetime import datetime

def setup_logging():
    """设置日志系统"""
    # 创建logs目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bfrm_realtime_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 设置日志配置
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*50)
    logger.info("BFRM Realtime v2.0.0 启动中...")
    logger.info("AI4CT Team - 气泡流重建与测量系统")
    logger.info("作者: BaodI Yu (yubaodi20@ipe.ac.cn)")
    logger.info("GitHub: https://github.com/AI4CT/BFRM")
    logger.info("="*50)
    
    try:
        # 创建Qt应用
        app = QApplication(sys.argv)
        app.setApplicationName("BFRM - Bubble Flow Reconstruction and Measurement")
        app.setApplicationVersion("2.0.0")
        app.setApplicationDisplayName("BFRM Realtime")
        app.setOrganizationName("AI4CT")
        app.setOrganizationDomain("ipe.ac.cn")
        
        # 设置应用图标（如果存在）
        icon_path = os.path.join("icons", "bfrm_icon.ico")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        
        # 设置默认字体
        font = QFont('Arial', 10)
        app.setFont(font)
        
        # 设置样式表（可选）
        app.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 10px 0px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        # 加载配置
        config = Config()
        
        # 创建主窗口
        main_window = RealtimeMainWindow(config)
        main_window.show()
        
        logger.info("BFRM Realtime启动完成")
        logger.info("系统准备就绪，等待用户操作...")
        
        # 运行应用
        return app.exec()
    
    except Exception as e:
        logger.error(f"启动失败: {str(e)}", exc_info=True)
        return -1
    
    finally:
        logger.info("BFRM Realtime已退出")

if __name__ == "__main__":
    sys.exit(main())

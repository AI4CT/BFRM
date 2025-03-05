# BFRM - Bubbly Flow Reconstruction Model

本软件用于将高速相机拍摄的气泡动态流场图像重建为3D的气泡结构。

## 功能介绍
1. 手动选择数据文件夹（默认路径：/home/yubd/mount/dataset/individual_bubble）。
2. 手动选择视频文件，支持常见格式如MP4、AVI、MOV和MKV。
3. 根据 CSV 筛选气泡图像，并根据气泡坐标参数对图像进行扩展和定位。
4. 对选择的视频文件进行逐帧处理和显示。
5. 生成连续气泡流图像的 GIF 和 MP4 动画，且实时显示。
6. 根据椭圆参数绘制气泡的矩形边框和拟合椭圆。
7. 进行气泡 3D 重建，显示交互式的 3D 气泡椭球模型。
8. 界面上同时显示四个动图：气泡流图像、带框图像、椭圆图像和 3D 重建图像。
9. 集成YOLO深度学习模型，实现高精度气泡检测。
10. 实时显示气泡信息，包括位置、尺寸、角度、速度和体积。
11. 支持CUDA加速和多线程处理，提高视频播放流畅度。

## 文件结构 
- `main.py`: 主程序入口
- `gui.py`: 图形用户界面实现
- `image_processor.py`: 图像处理模块
- `video_processor.py`: 视频处理模块
- `ellipse_fitting.py`: 椭圆拟合算法
- `reconstruction3d.py`: 3D重建模块
- `animation_exporter.py`: 动画导出工具
- `inference_obb.py`: YOLO推理模块
- `requirements.txt`: 依赖库列表
- `data/`: 示例数据目录
- `model/`: 预训练模型目录
- `icons/`: 界面图标资源
- `results/`: 处理结果保存目录

## 安装与使用
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 安装CUDA支持（可选，但推荐用于加速）
```bash
# 安装CUDA Toolkit (根据您的GPU选择合适的版本)
# 访问 https://developer.nvidia.com/cuda-downloads 获取安装指南

# 安装支持CUDA的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. 下载YOLO预训练模型
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

4. 运行程序
```bash
python main.py
```

## 操作指南
1. 启动程序后，可以通过"选择视频"按钮加载视频文件。
2. 视频加载后，可以使用播放控制工具栏控制视频播放。
3. 气泡信息会实时显示在右侧信息面板中。
4. 可以通过滑动条调整播放位置。
5. 可以通过"导出视频"按钮将处理后的视频保存。

## 性能优化
1. CUDA加速
   - 自动检测CUDA可用性，优先使用GPU进行YOLO模型推理
   - 支持在状态栏显示当前使用的设备（CPU/GPU）

2. 多线程优化
   - 使用线程池进行帧预处理，提高播放流畅度
   - 实现帧缓存机制，减少重复计算
   - 智能缓存管理，自动清理不需要的缓存

## 更新日志

### 2025年3月5日更新 (v1.2.0)
1. 性能优化
   - 添加CUDA加速支持，提高YOLO模型推理速度
   - 实现多线程预处理，提高视频播放流畅度
   - 添加智能帧缓存机制，减少重复计算

2. 界面优化
   - 移除了视频信息窗口，使界面更加简洁
   - 为气泡信息区域添加滚动条功能，支持大量气泡信息显示
   - 优化了气泡信息表格样式，添加行号列
   - 修复了字体警告问题

3. 功能改进
   - 视频循环播放时自动清空气泡轨迹，避免轨迹混乱
   - 气泡数量显示改为实时参数
   - 优化了表格头部固定功能，方便查看列名
   - 状态栏显示缓存状态和设备信息

### 2025年2月25日更新 (v1.1.0)
1. 集成YOLO检测模型
   - 添加实时气泡检测功能
   - 支持气泡参数自动计算

2. 界面改进
   - 添加视频控制工具栏
   - 支持播放速度调整
   - 添加循环播放功能

## 开发团队
- 于宝地 (主要开发者)
- AI4CT团队

## 许可证
© 2025 AI4CT团队，保留所有权利 
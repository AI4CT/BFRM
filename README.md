# BFRM - 气泡流重建与测量系统

<div align="center">
<img src="icons/app_icon.png" alt="BFRM Logo" width="150"/>

气泡流检测 | 追踪 | 分析 | 三维重建
</div>

## 📖 项目介绍

BFRM (Bubble Flow Reconstruction and Measurement) 是一套面向气泡流研究的软件系统，集成了气泡检测、追踪、分析与三维重建功能。系统通过计算机视觉和深度学习技术，实现对气泡流的各项物理参数精确测量与分析，为气泡流研究提供可靠的实验工具。

### 🔍 主要功能

- **气泡检测**：基于YOLOv8的高精度气泡检测
- **多目标追踪**：采用BotSort算法实现气泡轨迹追踪
- **数据分析**：气泡数量、尺寸、速度等参数统计分析
- **三维重建**：基于二维图像的气泡流三维空间结构重建
- **可视化**：丰富的2D/3D可视化呈现方式
- **气泡区域提取**：自动提取正方形气泡区域，边缘气泡自动白色填充

## 🚀 安装指南

### 系统要求

- Python 3.8+
- CUDA 11.0+ (用于GPU加速，推荐)
- 8GB+ 内存
- Windows 10/11, Linux 或 macOS

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/AI4CT/BFRM.git
cd BFRM
```

2. 创建并激活虚拟环境 (可选但推荐)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 如需GPU加速，确保已正确安装CUDA和cuDNN

## 🎮 使用指南

### 启动程序

```bash
python main.py
```

### 基本工作流程

1. **加载数据**：通过GUI界面加载气泡流视频
2. **运行检测**：使用YOLOv8模型检测气泡位置
3. **气泡追踪**：应用多目标追踪算法追踪气泡运动轨迹
4. **数据分析**：分析气泡数量、尺寸、轨迹等信息
5. **三维重建**：基于检测和追踪结果重建气泡流三维结构
6. **结果导出**：将分析结果导出为视频、图像或数据文件

### 输出文件结构

- `bubble_csv/`：气泡检测结果CSV文件
- `detection_with_tracks.mp4`：可视化气泡追踪结果
- `analysis_results/`：分析结果与统计图表
  - `bubble_crops/`：提取的气泡区域图像（正方形，边缘白色填充）
  - `visualizations/`：气泡轨迹和检测结果可视化
- `bubble_info/`：三维重建信息
- `visualizations/`：三维可视化结果

## 📊 软件架构

```
BFRM
│
├── GUI模块 ─────────────── 用户界面与交互
│   │
│   ├── 视频加载与播放
│   ├── 参数设置
│   ├── 结果可视化
│   └── 导出功能
│
├── 检测与追踪模块 ──────── 气泡识别与追踪
│   │
│   ├── YOLOv8检测
│   ├── BotSort追踪
│   └── GPU加速
│
├── 分析模块 ─────────────── 数据处理与统计
│   │
│   ├── 气泡数量分析
│   ├── 轨迹分析
│   ├── 气泡区域提取（正方形，边缘白色填充）
│   └── 统计可视化
│
└── 三维重建模块 ──────────── 空间结构重建
    │
    ├── 椭球体模型拟合
    ├── 空间结构初始化
    ├── 结构更新(计划中)
    └── 三维可视化
```

## 💻 开发状态

### 已完成功能

- [x] GUI界面开发
- [x] 气泡检测与追踪
- [x] 气泡信息统计与分析
- [x] 气泡区域提取（正方形，边缘白色填充）
- [x] 第一帧三维结构初始化
- [x] 基础三维可视化

### 开发中功能

- [ ] 后续帧三维结构更新
- [ ] 气泡流在线测量与分析
- [ ] 气泡流特性预测
- [ ] BubSort多目标追踪算法优化
- [ ] 多视角三维重建

## 🔧 技术栈

- **UI**: PyQt6
- **计算机视觉**: OpenCV, YOLOv8
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib, Plotly, PyVista
- **深度学习**: PyTorch
- **加速**: CUDA, GPU

## 🤝 贡献指南

欢迎对BFRM项目做出贡献！请参阅[贡献指南](CONTRIBUTING.md)了解详情。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 📬 联系方式

如有问题或建议，请联系项目维护者或提交Issue。

---

<div align="center">
© 2024 BFRM Team. All Rights Reserved.
</div>

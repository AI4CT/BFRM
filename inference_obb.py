from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QTableWidgetItem
import sys

def track(label=False):
    model = YOLO(r'C:\codebase\yolo\model\yolo11n-obb.pt')
    results = model.track(r'C:\codebase\yolo\image\4.mp4', stream=True, save=True, show_boxes=True, tracker='botsort.yaml')
    trajectories = {}  # 存储每个气泡的轨迹

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(r'C:\codebase\yolo\runs/tracked_bubbles.mp4', fourcc, 30.0, (1280, 800))  # 根据需要调整分辨率

    # 创建GUI窗口
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    window.setWindowTitle("气泡信息")
    layout = QtWidgets.QVBoxLayout()

    table = QtWidgets.QTableWidget()
    table.setColumnCount(9)
    table.setHorizontalHeaderLabels(["气泡编号", "横坐标", "纵坐标", "宽", "高", "旋转角度", "速度 (m/s)", "加速度", "体积 (mm³)"])
    layout.addWidget(table)

    # 新增 QLabel 用于显示检测结果图像
    image_label = QtWidgets.QLabel()
    layout.addWidget(image_label)

    window.setLayout(layout)  # 确保在添加表格后设置布局
    window.show()  # 在处理之前显示窗口

    for frame, r in enumerate(results):
        im = r.orig_img.copy()
        
        lw = max(round(sum(im.shape) / 2 * 0.002), 2)
        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3
        txt_color=(255, 255, 255)

        # 清空表格以更新信息
        table.setRowCount(0)

        for i in range(len(r.obb)):
            box = r.obb.xyxyxyxy[i].reshape(-1, 4, 2).squeeze()
            rbox = r.obb.xywhr[i]
            center = (rbox[0], rbox[1])  # 使用气泡的中心位置，保持原始数据
            p1 = [int(b) for b in box[0]]

            # 更新气泡轨迹
            bubble_id = int(r.obb.id[i])
            if bubble_id not in trajectories:
                trajectories[bubble_id] = []
            trajectories[bubble_id].append(center)
            if len(trajectories[bubble_id]) > 100:
                trajectories[bubble_id].pop(0)  # 保留最近的30帧

            color = (0, 0, 192) if r.obb.cls[i] else (53, 130, 84)
            # 绘制轨迹
            for j in range(1, len(trajectories[bubble_id])):
                # 确保轨迹点是整数类型
                start_point = tuple(map(int, trajectories[bubble_id][j-1]))
                end_point = tuple(map(int, trajectories[bubble_id][j]))
                cv2.line(im, start_point, end_point, color, 2)

            cv2.polylines(im, [np.asarray(box, dtype=int)], True, color, lw)  # cv2 requires nparray box
            if label:
                label = f"id:{bubble_id} {r.names[int(r.obb.cls[i])]}"
                w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
                h += 3  # add pixels to pad text
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > im.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                    p1 = im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    sf,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )
            # 计算气泡信息
            w, h = rbox[2], rbox[3]
            angle = rbox[4] * 180 / np.pi  # 角度转换
            
            # 检查轨迹长度以避免 IndexError
            if len(trajectories[bubble_id]) >= 2:
                speed = np.linalg.norm(np.array(trajectories[bubble_id][-1]) - np.array(trajectories[bubble_id][-2])) * 0.080128 * 0.001 / 0.00001  # 速度计算
            else:
                speed = 0  # 如果轨迹不足两个点，速度设为0
            
            volume = (w * w * h * 0.080128 ** 3)  # 体积计算
            
            # 更新表格，使用原始数据
            row_position = table.rowCount()
            table.insertRow(row_position)
            table.setItem(row_position, 0, QTableWidgetItem(str(bubble_id)))
            table.setItem(row_position, 1, QTableWidgetItem(f"{center[0]:.2f}"))
            table.setItem(row_position, 2, QTableWidgetItem(f"{center[1]:.2f}"))
            table.setItem(row_position, 3, QTableWidgetItem(f"{w:.2f}"))
            table.setItem(row_position, 4, QTableWidgetItem(f"{h:.2f}"))
            table.setItem(row_position, 5, QTableWidgetItem(f"{angle:.2f}"))
            table.setItem(row_position, 6, QTableWidgetItem(f"{speed:.2f}"))
            table.setItem(row_position, 7, QTableWidgetItem("N/A"))  # 加速度暂时未计算
            table.setItem(row_position, 8, QTableWidgetItem(f"{volume:.2f}"))

        # 更新 QLabel 显示当前帧图像
        height, width, channel = im.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(im.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        image_label.setPixmap(QtGui.QPixmap.fromImage(q_img))

        # 刷新窗口以显示更新
        app.processEvents()  # 处理事件以更新窗口
        img_path = f'{r.save_dir}/frame_{frame:03d}.jpg'
        # 保存图片
        cv2.imwrite(img_path, np.asarray(im))
        # 写入视频帧
        out.write(im)

    # 释放视频写入器
    out.release()
    print(f'图片已保存到: {img_path}')

    # 显示气泡流界面的气含率和气泡数量
    gas_content = (len(trajectories) * (0.080128 ** 2)) / (1280 * 800)  # 假设气泡流界面为1280x800
    print(f'气泡数量: {len(trajectories)}, 气含率: {gas_content:.4f}')

    sys.exit(app.exec())

if __name__ == '__main__':
    track()
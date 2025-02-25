import cv2
import numpy as np

def draw_rotated_rectangle(image, center, width, height, angle):
    """
    在图像上绘制旋转矩形。
    
    Args:
        image: 输入图像
        center: 矩形中心点 (x, y)
        width: 矩形宽度
        height: 矩形高度
        angle: 旋转角度（度）
    """
    # 计算矩形的四个顶点
    angle_rad = np.deg2rad(angle)  # 保持原始角度方向
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    half_w = width / 2
    half_h = height / 2
    
    # 计算旋转后的四个顶点
    points = []
    for x, y in [(-half_w, -half_h), (half_w, -half_h), 
                    (half_w, half_h), (-half_w, half_h)]:
        new_x = x * cos_theta - y * sin_theta + center[0]
        new_y = x * sin_theta + y * cos_theta + center[1]
        points.append([int(new_x), int(new_y)])
    
    points = np.array(points)
    
    # 绘制矩形
    cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
    return image

def draw_bounding_box_on_image(frame, data_row=None, bubble_history=None):
    """
    在图像上绘制旋转边界框和气泡轨迹。
    使用CSV文件中的w_r、h_r、theta参数。
    
    Args:
        frame: 输入图像
        data_row: 当前帧的气泡数据
        bubble_history: 气泡历史轨迹数据列表
    """
    if data_row is None:
        return frame
        
    img = frame.copy()
    # 添加64像素的偏移
    center = (int(data_row['x0']) + 64, int(data_row['y0']) + 64)
    width = float(data_row['w_r'])
    height = float(data_row['h_r'])
    angle = -float(data_row['theta'])  # 修正角度方向，与椭圆保持一致
    
    # 绘制边界框
    img = draw_rotated_rectangle(img, center, width, height, angle)
    
    # 绘制历史轨迹
    if bubble_history is not None and len(bubble_history) > 1:
        # 计算最大速度用于颜色映射
        max_speed = 0
        dt = 0.0001  # 帧间隔时间（秒）
        pixel_scale = 0.080128  # 像素尺度（毫米/像素）
        
        speeds = []  # 存储所有速度值
        for i in range(1, len(bubble_history)):
            dx = (bubble_history[i]['x0'] - bubble_history[i-1]['x0']) * pixel_scale
            dy = (bubble_history[i]['y0'] - bubble_history[i-1]['y0']) * pixel_scale
            speed = np.sqrt(dx**2 + dy**2) / dt  # mm/s
            speed_ms = speed / 1000  # 转换为m/s
            speeds.append(speed_ms)
            max_speed = max(max_speed, speed_ms)
        
        # print(f"轨迹点数: {len(bubble_history)}, 最大速度: {max_speed:.3f} m/s")
        
        # 绘制历史轨迹点和线
        for i in range(1, len(bubble_history)):
            prev_center = (int(bubble_history[i-1]['x0']) + 64, int(bubble_history[i-1]['y0']) + 64)
            curr_center = (int(bubble_history[i]['x0']) + 64, int(bubble_history[i]['y0']) + 64)
            
            # 计算当前速度
            dx = (bubble_history[i]['x0'] - bubble_history[i-1]['x0']) * pixel_scale
            dy = (bubble_history[i]['y0'] - bubble_history[i-1]['y0']) * pixel_scale
            speed = np.sqrt(dx**2 + dy**2) / dt  # mm/s
            speed_ms = speed / 1000  # 转换为m/s
            
            # 根据速度计算颜色（红色表示快，蓝色表示慢）
            ratio = speed_ms / max_speed if max_speed > 0 else 0
            color = (int(255 * (1 - ratio)), 0, int(255 * ratio))  # BGR格式
            
            # 绘制轨迹线
            cv2.line(img, prev_center, curr_center, color, 1)
            
            # 绘制轨迹点
            cv2.circle(img, curr_center, 2, color, -1)
    
    return img

def draw_ellipse_on_image(frame, data_row=None, bubble_history=None):
    """
    在图像上绘制椭圆和气泡轨迹。
    使用CSV文件中的w_r、h_r、theta参数。
    
    Args:
        frame: 输入图像
        data_row: 当前帧的气泡数据
        bubble_history: 气泡历史轨迹数据列表
    """
    if data_row is None:
        return frame
        
    img = frame.copy()
    # 添加64像素的偏移
    center = (int(data_row['x0']) + 64, int(data_row['y0']) + 64)
    axes = (int(data_row['w_r']/2), int(data_row['h_r']/2))  # 椭圆轴长为矩形边长的一半
    angle = -float(data_row['theta'])  # 修正角度方向
    
    # 绘制当前气泡
    cv2.ellipse(img, center, axes, angle, 0, 360, (0, 0, 255), 2)
    
    # 绘制历史轨迹
    if bubble_history is not None and len(bubble_history) > 1:  # 确保至少有两个点才能绘制轨迹
        # 计算最大速度用于颜色映射
        max_speed = 0
        dt = 0.0001  # 帧间隔时间（秒）
        pixel_scale = 0.080128  # 像素尺度（毫米/像素）
        
        speeds = []  # 存储所有速度值
        for i in range(1, len(bubble_history)):
            dx = (bubble_history[i]['x0'] - bubble_history[i-1]['x0']) * pixel_scale
            dy = (bubble_history[i]['y0'] - bubble_history[i-1]['y0']) * pixel_scale
            speed = np.sqrt(dx**2 + dy**2) / dt  # mm/s
            speed_ms = speed / 1000  # 转换为m/s
            speeds.append(speed_ms)
            max_speed = max(max_speed, speed_ms)
        
        # print(f"轨迹点数: {len(bubble_history)}, 最大速度: {max_speed:.3f} m/s")
        
        # 绘制历史轨迹点和线
        for i in range(1, len(bubble_history)):
            prev_center = (int(bubble_history[i-1]['x0']) + 64, int(bubble_history[i-1]['y0']) + 64)
            curr_center = (int(bubble_history[i]['x0']) + 64, int(bubble_history[i]['y0']) + 64)
            
            # 计算当前速度
            dx = (bubble_history[i]['x0'] - bubble_history[i-1]['x0']) * pixel_scale
            dy = (bubble_history[i]['y0'] - bubble_history[i-1]['y0']) * pixel_scale
            speed = np.sqrt(dx**2 + dy**2) / dt  # mm/s
            speed_ms = speed / 1000  # 转换为m/s
            
            # 根据速度计算颜色（红色表示快，蓝色表示慢）
            ratio = speed_ms / max_speed if max_speed > 0 else 0
            color = (int(255 * (1 - ratio)), 0, int(255 * ratio))  # BGR格式
            
            # 绘制轨迹线，增加线条粗细
            cv2.line(img, prev_center, curr_center, color, 1)
            
            # 绘制轨迹点，增加点的大小
            cv2.circle(img, curr_center, 2, color, -1)
    
    return img
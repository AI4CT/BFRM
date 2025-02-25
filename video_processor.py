import os
import cv2
import numpy as np

def validate_video(video_path):
    """验证视频文件的有效性"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"无法读取视频: {video_path}")
    cap.release()
    return True

def get_video_info(video_path):
    """获取视频的基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"无法读取视频: {video_path}")
    
    # 获取视频的基本信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': frame_count / fps if fps > 0 else 0
    }

def process_video(video_path, output_folder, progress_callback=None):
    """
    处理视频文件，将视频逐帧读取并转换为图像序列
    
    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹路径
        progress_callback: 进度回调函数
    
    Returns:
        tuple: (frames, frame_count, fps) - 处理后的帧列表、总帧数和帧率
    """
    try:
        # 验证视频文件
        validate_video(video_path)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频总帧数: {frame_count}, FPS: {fps}")
        
        # 读取视频帧
        all_frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 如果需要，这里可以添加对帧的处理逻辑，例如调整大小
            all_frames.append(frame)
            
            # 更新进度
            if progress_callback and frame_count > 0:
                progress_callback(int(100 * (i + 1) / frame_count))
        
        cap.release()
        
        print(f"成功处理了 {len(all_frames)} 帧视频")
        return all_frames, len(all_frames), fps
        
    except Exception as e:
        error_msg = f"处理视频时出错: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg) 
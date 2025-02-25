import os
import imageio
import cv2
import numpy as np

def export_animation(frames, output_folder, progress_callback=None):
    """
    将连续的气泡流图像序列导出为 GIF 与 MP4 文件。
    设置帧率为30fps。
    """
    gif_path = os.path.join(output_folder, "bubbly_flow.gif")
    mp4_path = os.path.join(output_folder, "bubbly_flow.mp4")
    
    total_frames = len(frames)
    fps = 30  # 设置帧率为30fps
    
    try:
        # 对于大规模视频，GIF导出可能会很慢
        # 这里我们可以考虑减少帧数或缩小分辨率
        if total_frames > 500:
            print(f"帧数较多 ({total_frames}), 对GIF进行采样处理...")
            # 采样率，每10帧取1帧
            sample_rate = max(1, total_frames // 300)
            sampled_frames = frames[::sample_rate]
            
            # 缩小分辨率
            resized_frames = []
            for frame in sampled_frames:
                h, w = frame.shape[:2]
                resized = cv2.resize(frame, (w//2, h//2))
                resized_frames.append(resized)
                
            print(f"GIF采样后帧数: {len(resized_frames)}")
            
            # GIF导出
            print("正在导出GIF动画...")
            imageio.mimsave(
                gif_path, 
                resized_frames, 
                fps=fps//2,
                optimize=True,
                subrectangles=True
            )
        else:
            # GIF导出
            print("正在导出GIF动画...")
            imageio.mimsave(
                gif_path, 
                frames, 
                fps=fps,
                optimize=True,
                subrectangles=True
            )
            
        print(f"GIF动画已保存至: {gif_path}")
        
        # MP4导出
        print("正在导出MP4视频...")
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编码器
        video_writer = cv2.VideoWriter(
            mp4_path, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        for i, frame in enumerate(frames):
            video_writer.write(frame)
            if progress_callback and i % 10 == 0:  # 每10帧更新一次进度
                progress_callback(int(100 * i / total_frames))
                
        video_writer.release()
        print(f"MP4视频已保存至: {mp4_path}")
        
        return gif_path, mp4_path
        
    except Exception as e:
        error_msg = f"导出动画时出错: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg) 
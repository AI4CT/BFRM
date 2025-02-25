import os
import pandas as pd
import cv2

def validate_image(img_path):
    """验证图像的有效性"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    if img.shape[:2] != (128, 128):
        raise ValueError(f"图像尺寸不正确: {img_path}")
    return True

def get_image_count(folder):
    """获取文件夹中有效图像的数量"""
    image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return len(image_files)

def group_bubbles(data, images):
    """
    根据帧号将气泡分组，连续帧的气泡归为一组。
    
    Args:
        data: DataFrame, 包含气泡信息的数据表
        images: list, 气泡图像列表
    
    Returns:
        list of tuples: [(group_data, group_images), ...], 每个元组包含一组连续帧的数据和图像
    """
    if data.empty or not images:
        return []
    
    # 确保数据按帧号排序
    data = data.sort_values('frame')
    groups = []
    current_group_data = []
    current_group_images = []
    last_frame = None
    
    # 遍历所有数据
    for idx, row in data.iterrows():
        if idx >= len(images):
            break
            
        current_frame = row['frame']
        if last_frame is None:
            # 第一个气泡
            current_group_data.append(row)
            current_group_images.append(images[idx])
        elif current_frame == last_frame + 1:
            # 连续帧
            current_group_data.append(row)
            current_group_images.append(images[idx])
        else:
            # 不连续，保存当前组并开始新组
            if current_group_data:
                groups.append((pd.DataFrame(current_group_data), current_group_images))
                current_group_data = [row]
                current_group_images = [images[idx]]
        
        last_frame = current_frame
    
    # 添加最后一组
    if current_group_data:
        groups.append((pd.DataFrame(current_group_data), current_group_images))
    
    return groups

def load_bubble_data(folder, preprocess=True):
    """
    加载指定文件夹下气泡图像和CSV文件数据。
    按照文件名顺序加载图像。
    
    Returns:
        tuple: (DataFrame, list[ndarray]) - (CSV数据, 图像列表)
    """
    try:
        # 加载CSV文件
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("未找到csv文件")
        
        csv_path = os.path.join(folder, csv_files[0])
        data = pd.read_csv(csv_path)
        
        # 按数字顺序排序图像文件
        image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        images = []
        for img_file in image_files:
            try:
                img_path = os.path.join(folder, img_file)
                validate_image(img_path)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图像: {img_file}")
                    continue
                images.append(img)
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {str(e)}")
                continue
        
        if not images:
            raise ValueError("没有成功加载任何图像")
            
        return data, images
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise 
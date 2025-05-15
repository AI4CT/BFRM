import torch
import torch.nn as nn

def chamfer_distance(x, y, eps=1e-8):
    """
    计算两组点云之间的Chamfer距离，增加数值稳定性
    x: 批次 x 点数1 x 3, 第一组点云
    y: 批次 x 点数2 x 3, 第二组点云
    eps: 数值稳定性的小常数
    返回: 点云间的距离以及索引
    """
    batch_size, n, _ = x.shape
    _, m, _ = y.shape
    
    # 检查输入是否包含NaN或Inf
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
    if torch.isnan(y).any() or torch.isinf(y).any():
        y = torch.nan_to_num(y, nan=0.0, posinf=1e5, neginf=-1e5)
    
    # 将点扩展为批次差异矩阵
    x_expanded = x.unsqueeze(2)  # [B, N, 1, 3]
    y_expanded = y.unsqueeze(1)  # [B, 1, M, 3]
    
    # 计算欧氏距离的平方，添加eps防止数值问题
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=3) + eps  # [B, N, M]
    
    # 使用clamp限制最大/最小值，防止溢出
    dist = torch.clamp(dist, min=eps, max=1e9)
    
    # 计算x到y的最小距离及索引
    min_dist_x_to_y, idx1 = torch.min(dist, dim=2)  # [B, N]
    
    # 计算y到x的最小距离及索引
    min_dist_y_to_x, idx2 = torch.min(dist, dim=1)  # [B, M]
    
    # 检查输出是否包含NaN
    if torch.isnan(min_dist_x_to_y).any() or torch.isnan(min_dist_y_to_x).any():
        print("警告：Chamfer距离计算产生了NaN值")
        min_dist_x_to_y = torch.nan_to_num(min_dist_x_to_y, nan=1.0)
        min_dist_y_to_x = torch.nan_to_num(min_dist_y_to_x, nan=1.0)
    
    return min_dist_x_to_y, min_dist_y_to_x, idx1, idx2


class ChamferDist(nn.Module):
    """
    Chamfer距离模块，包装了chamfer_distance函数作为PyTorch模块
    """
    def __init__(self):
        super(ChamferDist, self).__init__()

    def forward(self, input1, input2):
        return chamfer_distance(input1, input2)

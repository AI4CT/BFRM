import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.chamfer_wrapper import ChamferDist


class P2MLoss(nn.Module):
    def __init__(self, options, ellipsoid):
        super().__init__()
        self.options = options
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()
        self.laplace_idx = nn.ParameterList([
            nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx])
        self.edges = nn.ParameterList([
            nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])
        self.faces = [nn.Parameter(faces, requires_grad=False) for faces in ellipsoid.faces]

    def edge_regularization(self, pred, edges):
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * pred.size(-1)

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """

        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        vertices = inputs[:, all_valid_indices]
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(self, input1, input2, block_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)
        move_loss = self.l2_loss(input1, input2) * input1.size(-1) if block_idx > 0 else 0
        return laplace_loss, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):
        edges = F.normalize(pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2)
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def calculate_volume(self, vertices, faces):
        """计算3D网格的体积
        
        Args:
            vertices: 形状为 (batch_size, num_vertices, 3) 的顶点坐标
            faces: 形状为 (num_faces, 3) 的面索引
            
        Returns:
            形状为 (batch_size,) 的体积值
        """
        # 获取面的三个顶点
        v0 = vertices[:, faces[:, 0]]
        v1 = vertices[:, faces[:, 1]]
        v2 = vertices[:, faces[:, 2]]
        
        # 计算每个四面体的有符号体积
        # 假设原点是参考点
        # 体积 = (v0 · (v1 × v2)) / 6
        cross = torch.cross(v1, v2, dim=2)
        vol = torch.sum(v0 * cross, dim=2) / 6.0
        
        # 计算总体积（取绝对值以防止负值）
        total_volume = torch.abs(torch.sum(vol, dim=1))
        return total_volume

    def volume_consistency_loss(self, pred_coords_list, gt_volume=None):
        """计算多组预测结果的体积一致性损失 - 简化版
        
        Args:
            pred_coords_list: 预测坐标列表，每个元素形状为 (batch_size, num_vertices, 3)
            gt_volume: 可选的真实体积，形状为 (batch_size,)
            
        Returns:
            loss: 体积一致性损失
            loss_dict: 损失详情字典
        """
        # 计算每个预测结果的体积
        volumes = []
        for pred_coords in pred_coords_list:
            volume = self.calculate_volume(pred_coords, self.faces[-1])
            volumes.append(volume)
        
        # 将体积堆叠为张量
        volumes = torch.stack(volumes, dim=1)  # (batch_size, num_predictions)
        
        # 体积一致性损失：体积的变异系数(标准差/均值)
        # 这可以确保相对的体积一致性，不受绝对体积大小的影响
        volume_mean = torch.mean(volumes, dim=1)  # (batch_size,)
        volume_std = torch.std(volumes, dim=1)  # (batch_size,)
        consistency_loss = torch.mean(volume_std / (volume_mean + 1e-8))
        
        # 如果提供了真实体积，添加真值匹配损失
        gt_error = 0.0
        if gt_volume is not None:
            gt_error = torch.mean(torch.abs(volume_mean - gt_volume) / (gt_volume + 1e-8))
            total_loss = consistency_loss + gt_error
        else:
            total_loss = consistency_loss
        
        # 返回损失和详细信息
        return total_loss, {
            "loss_volume_total": total_loss.item(),
            "loss_volume_consistency": consistency_loss.item(),
            "loss_volume_gt_error": gt_error if isinstance(gt_error, float) else gt_error.item(),
            "mean_volume": volume_mean.mean().item(),
            "volume_std": volume_std.mean().item()
        }

    def temporal_consistency_loss(self, outputs_sequence, gt_volume=None):
        """计算时间一致性损失 - 自监督版本，只基于体积一致性
        
        Args:
            outputs_sequence: 包含模型在序列中每一帧的输出
            gt_volume: 真值体积，对于时间一致性不使用该参数
            
        Returns:
            loss: 时间一致性损失
            loss_summary: 损失摘要
        """
        # 提取最终层的预测坐标
        pred_coords_sequence = [output["pred_coord"][-1] for output in outputs_sequence]
        
        # 直接计算每个时间点预测结果的体积
        volumes = []
        for pred_coords in pred_coords_sequence:
            volume = self.calculate_volume(pred_coords, self.faces[-1])
            volumes.append(volume)
        
        # 将体积堆叠为张量
        volumes = torch.stack(volumes, dim=1)  # (batch_size, sequence_length)
        
        # 计算体积一致性损失（自监督部分）
        volume_mean = torch.mean(volumes, dim=1)  # (batch_size,)
        volume_std = torch.std(volumes, dim=1)  # (batch_size,)
        consistency_loss = torch.mean(volume_std / (volume_mean + 1e-8))
        
        # 返回损失和详细信息
        loss_dict = {
            "loss_temporal_total": consistency_loss.item(),
            "loss_temporal_consistency": consistency_loss.item(),
            "mean_volume": volume_mean.mean().item(),
            "volume_std": volume_std.mean().item()
        }
        
        return consistency_loss, loss_dict

    def spatial_consistency_loss(self, outputs_multiview, gt_volume=None):
        """计算空间一致性损失 - 多视角重建体积一致性
        
        Args:
            outputs_multiview: 包含模型在不同视角的输出列表
            gt_volume: 真实体积，形状为(batch_size,)
            
        Returns:
            loss: 空间一致性损失
            loss_summary: 损失摘要
        """
        # 获取权重
        consistency_weight = getattr(self.options.weights, "spatial_consistency_weight", 0.5)
        gt_weight = getattr(self.options.weights, "spatial_gt_weight", 0.5)
        
        # 提取最终层的预测坐标
        pred_coords_multiview = [output["pred_coord"][-1] for output in outputs_multiview]
        
        # 直接计算每个视角预测结果的体积，无需旋转
        volumes = []
        for pred_coords in pred_coords_multiview:
            volume = self.calculate_volume(pred_coords, self.faces[-1])
            volumes.append(volume)
        
        # 将体积堆叠为张量
        volumes = torch.stack(volumes, dim=1)  # (batch_size, num_views)
        
        # 1. 计算体积一致性损失（自监督部分）
        # 使用变异系数(CV)作为一致性指标 - 这是标准差除以均值
        volume_mean = torch.mean(volumes, dim=1)  # (batch_size,)
        volume_std = torch.std(volumes, dim=1)  # (batch_size,)
        consistency_loss = torch.mean(volume_std / (volume_mean + 1e-8))
        
        # 2. 计算与真值体积的匹配度（有监督部分）
        gt_error = 0.0
        if gt_volume is not None:
            # 计算所有视角平均体积与真值体积的相对误差
            gt_error = torch.mean(torch.abs(volume_mean - gt_volume) / (gt_volume + 1e-8))
            
            # 同时确保每个视角的重建体积都接近真值
            per_view_gt_error = torch.mean(torch.abs(volumes - gt_volume.unsqueeze(1)) / (gt_volume.unsqueeze(1) + 1e-8))
            gt_error = 0.5 * (gt_error + per_view_gt_error)
        
        # 3. 总空间一致性损失
        total_loss = consistency_weight * consistency_loss
        if gt_volume is not None:
            total_loss += gt_weight * gt_error
        
        # 计算各个视角体积的统计信息以便调试
        min_vol = torch.min(volumes).item()
        max_vol = torch.max(volumes).item()
        
        # 返回损失和详细信息
        loss_dict = {
            "loss_spatial_total": total_loss.item(),
            "loss_spatial_consistency": consistency_loss.item(),
            "loss_spatial_gt_error": gt_error if isinstance(gt_error, float) else gt_error.item(),
            "mean_volume": volume_mean.mean().item(),
            "volume_std": volume_std.mean().item(),
            "min_volume": min_vol,
            "max_volume": max_vol
        }
        
        return total_loss, loss_dict

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """

        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss = 0., 0., 0., 0., 0.
        lap_const = [0.2, 1., 1.]

        gt_coord, gt_normal, gt_images = targets["points"], targets["normals"], targets["images"]
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]
        image_loss = 0.
        if outputs["reconst"] is not None and self.options.weights.reconst != 0:
            image_loss = self.image_loss(gt_images, outputs["reconst"])

        for i in range(3):
            dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, pred_coord[i])
            chamfer_loss += self.options.weights.chamfer[i] * (torch.mean(dist1) +
                                                               self.options.weights.chamfer_opposite * torch.mean(dist2))
            normal_loss += self.normal_loss(gt_normal, idx2, pred_coord[i], self.edges[i])
            edge_loss += self.edge_regularization(pred_coord[i], self.edges[i])
            lap, move = self.laplace_regularization(pred_coord_before_deform[i],
                                                                    pred_coord[i], i)
            lap_loss += lap_const[i] * lap
            move_loss += lap_const[i] * move

        loss = chamfer_loss + image_loss * self.options.weights.reconst + \
               self.options.weights.laplace * lap_loss + \
               self.options.weights.move * move_loss + \
               self.options.weights.edge * edge_loss + \
               self.options.weights.normal * normal_loss

        loss = loss * self.options.weights.constant

        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            "loss_move": move_loss,
            "loss_normal": normal_loss,
        }

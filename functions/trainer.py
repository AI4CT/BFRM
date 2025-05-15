import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from functions.evaluator import Evaluator
from models.classifier import Classifier
from models.losses.classifier import CrossEntropyLoss
from models.losses.p2m import P2MLoss
from models.p2m import P2MModel
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.tensor import recursive_detach
# from utils.vis.renderer import MeshRenderer

from datasets.shapenet import ShapeNet
from datasets.temporal_dataset import TemporalDataset
from datasets.spatial_dataset import SpatialDataset


class Trainer(CheckpointRunner):

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        # if self.options.model.name == "pixel2mesh":
        #     # Visualization renderer
        #     self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
        #                                  self.options.dataset.mesh_pos)
        #     # create ellipsoid
        #     self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        # else:
        #     self.renderer = None
        self.renderer = None
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh":
                # create model
                self.model = P2MModel(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)
            elif self.options.model.name == "classifier":
                self.model = Classifier(self.options.model, self.options.dataset.num_classes)
            else:
                raise NotImplementedError("Your model is not found")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        # Setup a joint optimizer for the 2 models
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif self.options.optim.name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        # Create loss functions
        if self.options.model.name == "pixel2mesh":
            self.criterion = P2MLoss(self.options.loss, self.ellipsoid).cuda()
        elif self.options.model.name == "classifier":
            self.criterion = CrossEntropyLoss()
        else:
            raise NotImplementedError("Your loss is not found")

        # Create AverageMeters for losses
        self.losses = AverageMeter()

        # Evaluators
        self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]

        # 创建三个数据集
        self.original_dataset = ShapeNet(
            root_dir=self.options.dataset.original_data_path, 
            file_list_name=None,  # 直接使用root_dir
            mesh_pos=self.options.dataset.mesh_pos, 
            normalization=self.options.dataset.normalization
        )
        self.temporal_dataset = TemporalDataset(
            self.options.dataset.temporal_data_path, 
            sequence_length=self.options.dataset.sequence_length,
            normalization=self.options.dataset.normalization
        )
        self.spatial_dataset = SpatialDataset(
            self.options.dataset.spatial_data_path,
            view_count=self.options.dataset.view_count,
            normalization=self.options.dataset.normalization,
        )
        
        # 创建三个数据加载器
        self.original_loader = None  # 将在train方法中每个epoch重新创建
        self.temporal_loader = None  # 将在train方法中每个epoch重新创建
        self.spatial_loader = None   # 将在train方法中每个epoch重新创建
        
        # 设置损失权重
        self.loss_weights = {
            'original': self.options.loss.original_weight,
            'temporal': self.options.loss.temporal_weight,
            'spatial': self.options.loss.spatial_weight
        }

    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        self.model.train()

        # Grab data from the batch
        images = input_batch["images"]

        # predict with model
        out = self.model(images)

        # compute loss
        loss, loss_summary = self.criterion(out, input_batch)
        self.losses.update(loss.detach().cpu().item())

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        # 检查是否有NaN或Inf的梯度
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"警告：参数 {name} 的梯度包含NaN或Inf，已将其替换为0")
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.optimizer.step()

        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def train_step_with_accumulated_gradients(self):
        """使用累计梯度方式训练所有数据集"""
        self.model.train()
        self.optimizer.zero_grad()  # 在所有数据集前向传播前清零梯度
        
        total_loss = 0
        loss_summary = {}
        
        # 1. 原始数据集
        if self.original_batch is not None:
            try:
                # 前向传播
                original_out = self.model(self.original_batch["images"])
                # 计算损失
                original_loss, original_loss_summary = self.criterion(original_out, self.original_batch)
                # 加权并累积梯度
                weighted_loss = self.loss_weights['original'] * original_loss
                weighted_loss.backward()
                # 更新总损失
                total_loss += weighted_loss.detach().cpu().item()
                # 记录损失
                for k, v in original_loss_summary.items():
                    loss_summary[f"original_{k}"] = v
                self.logger.debug(f"原始数据集损失: {original_loss.item():.6f}")
            except Exception as e:
                self.logger.error(f"处理原始数据集时出错: {str(e)}")
        
        # 2. 时间一致性数据集 (自监督)
        if self.temporal_batch is not None:
            try:
                # 处理序列数据
                batch_size, seq_len = self.temporal_batch["image_sequence"].shape[:2]
                outputs_sequence = []
                
                # 对序列中的每帧单独前向传播
                for i in range(seq_len):
                    current_image = self.temporal_batch["image_sequence"][:, i]
                    current_out = self.model(current_image)
                    outputs_sequence.append(current_out)
                
                # 计算时间一致性损失 (自监督，不使用真值)
                temporal_loss, temporal_loss_summary = self.criterion.temporal_consistency_loss(
                    outputs_sequence)
                
                # 加权并累积梯度
                weighted_loss = self.loss_weights['temporal'] * temporal_loss
                weighted_loss.backward()
                
                # 更新总损失
                total_loss += weighted_loss.detach().cpu().item()
                
                # 记录损失
                for k, v in temporal_loss_summary.items():
                    loss_summary[f"{k}"] = v
                self.logger.debug(f"时间一致性损失: {temporal_loss.item():.6f}")
            except Exception as e:
                self.logger.error(f"处理时间一致性数据集时出错: {str(e)}")
        
        # 3. 空间一致性数据集 (包括真值约束)
        if self.spatial_batch is not None:
            try:
                # 处理多视角数据
                batch_size, num_views = self.spatial_batch["multi_view_images"].shape[:2]
                outputs_multiview = []
                
                # 对每个视角单独前向传播
                for i in range(num_views):
                    current_image = self.spatial_batch["multi_view_images"][:, i]
                    current_out = self.model(current_image)
                    outputs_multiview.append(current_out)
                
                # 获取必要的数据
                gt_volume = self.spatial_batch["volume"].cuda() if "volume" in self.spatial_batch else None
                
                # 计算空间一致性损失
                spatial_loss, spatial_loss_summary = self.criterion.spatial_consistency_loss(
                    outputs_multiview,
                    gt_volume)
                
                # 加权并累积梯度
                weighted_loss = self.loss_weights['spatial'] * spatial_loss
                weighted_loss.backward()
                
                # 更新总损失
                total_loss += weighted_loss.detach().cpu().item()
                
                # 记录损失
                for k, v in spatial_loss_summary.items():
                    loss_summary[f"{k}"] = v
                
                # 额外记录空间一致性数据的详细信息
                if gt_volume is not None:
                    self.logger.debug(f"空间一致性损失: {spatial_loss.item():.6f}, 真实体积: {gt_volume.mean().item():.6f}")
                else:
                    self.logger.debug(f"空间一致性损失: {spatial_loss.item():.6f}")
                
                # 记录当前样本信息
                self.logger.debug(f"当前空间一致性样本: {self.spatial_batch['filename']}")
            except Exception as e:
                self.logger.error(f"处理空间一致性数据集时出错: {str(e)}")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        # 检查NaN或Inf梯度
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning(f"警告：参数 {name} 的梯度包含NaN或Inf，已将其替换为0")
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 执行优化步骤
        self.optimizer.step()
        
        # 返回总损失
        self.losses.update(total_loss)
        return loss_summary

    def train_summaries(self, input_batch, out_summary, loss_summary):
        # 确保input_batch不为None
        if input_batch is None:
            self.logger.warning("train_summaries收到了None类型的input_batch")
            return
            
        if self.renderer is not None:
            # Do visualization for the first 2 images of the batch
            try:
                render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
                self.summary_writer.add_image("render_mesh", render_mesh, self.step_count)
            except Exception as e:
                self.logger.error(f"生成3D渲染可视化时出错: {str(e)}")
                
            # 添加length分布图，如果存在
            if "length" in input_batch:
                try:
                    self.summary_writer.add_histogram("length_distribution", input_batch["length"].cpu().numpy(),
                                                self.step_count)
                except Exception as e:
                    self.logger.error(f"添加length分布图时出错: {str(e)}")

        # Debug info for filenames
        self.logger.debug(input_batch["filename"])

        # Save results in Tensorboard
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

        # Save results to log
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.9f (%.9f)" % (
            self.epoch_count, self.step_count,
            # self.options.train.num_epochs * len(self.dataset) // (
            #             self.options.train.batch_size * self.options.num_gpus),
            self.options.train.num_epochs * self.max_iterations // (
                        self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))
        
    def train(self):
        # 运行训练 num_epochs 个周期
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            # 为每个周期创建新的数据加载器
            self.original_loader = DataLoader(
                self.original_dataset,
                batch_size=self.options.train.batch_size * self.options.num_gpus,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=self.options.train.shuffle,
                collate_fn=self.dataset_collate_fn
            )
            
            self.temporal_loader = DataLoader(
                self.temporal_dataset,
                batch_size=self.options.train.temporal_batch_size * self.options.num_gpus,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=self.options.train.shuffle
            )
            
            self.spatial_loader = DataLoader(
                self.spatial_dataset,
                batch_size=self.options.train.spatial_batch_size * self.options.num_gpus,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=self.options.train.shuffle
            )

            # 创建数据迭代器
            original_iter = iter(self.original_loader)
            temporal_iter = iter(self.temporal_loader)
            spatial_iter = iter(self.spatial_loader)
            
            # 计算最大迭代次数 (使用最大的数据集长度)
            max_iterations = max(
                len(self.original_loader),
                len(self.temporal_loader),
                len(self.spatial_loader)
            )
            self.logger.info(f"最大迭代次数: {max_iterations}")
            self.max_iterations = max_iterations

            # 重置损失
            self.losses.reset()

            # 遍历所有批次
            for step in range(max_iterations):
                # 获取各数据集的批次
                try:
                    self.original_batch = next(original_iter)
                    self.original_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                                            for k, v in self.original_batch.items()}
                except StopIteration:
                    self.original_batch = None
                    
                try:
                    self.temporal_batch = next(temporal_iter)
                    self.temporal_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                                            for k, v in self.temporal_batch.items()}
                except StopIteration:
                    self.temporal_batch = None
                    
                try:
                    self.spatial_batch = next(spatial_iter)
                    self.spatial_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.spatial_batch.items()}
                except StopIteration:
                    self.spatial_batch = None
                
                # 如果所有数据集都耗尽，则跳出循环
                if self.original_batch is None and self.temporal_batch is None and self.spatial_batch is None:
                    break
                
                # 使用累计梯度方式进行训练
                loss_summary = self.train_step_with_accumulated_gradients()

                self.step_count += 1

                # 每 summary_steps 步在Tensorboard中记录
                if self.step_count % self.options.train.summary_steps == 0:
                    # 选择一个非空批次用于可视化
                    vis_batch = None
                    if self.original_batch is not None:
                        vis_batch = self.original_batch
                    elif self.temporal_batch is not None:
                        vis_batch = self.temporal_batch
                    elif self.spatial_batch is not None:
                        vis_batch = self.spatial_batch
                    
                    # 确保vis_batch不为None
                    if vis_batch is not None:
                        self.train_summaries(vis_batch, None, loss_summary)
                    else:
                        self.logger.warning("所有数据批次为None，跳过可视化")

            # 每个周期结束后保存检查点
            if self.epoch_count % self.options.train.test_epochs == 0:
                self.dump_checkpoint()

            # 每 test_epochs 运行一次验证
            if self.epoch_count % self.options.train.test_epochs == 0:
                self.test()

            # 学习率调度器步骤
            self.lr_scheduler.step()

    def test(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()

import os
from logging import Logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from functions.base import CheckpointRunner
from models.p2m import P2MModel
from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer
from options import update_options, options, reset_options
import sys
import argparse

class Predictor(CheckpointRunner):

    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        self.gpu_inference = self.options.num_gpus > 0
        if self.gpu_inference == 0:
            raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                      "might be fixed in the future.")

        if self.options.model.name == "pixel2mesh":
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
            # create model
            self.model = P2MModel(self.options.model, self.ellipsoid,
                                  self.options.dataset.camera_f, self.options.dataset.camera_c,
                                  self.options.dataset.mesh_pos)
            if self.gpu_inference:
                self.model.cuda()
                # create renderer
                self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                             self.options.dataset.mesh_pos)
        else:
            raise NotImplementedError("Currently the predictor only supports pixel2mesh")

    def models_dict(self):
        return {'model': self.model}

    def predict_step(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            images = input_batch['images']
            out = self.model(images)
            self.save_inference_results(input_batch, out)

    def predict(self):
        self.logger.info("Running predictions...")
        predict_data_loader = DataLoader(self.dataset,
                                            batch_size=self.options.test.batch_size,
                                            pin_memory=self.options.pin_memory,
                                            collate_fn=self.dataset_collate_fn)

        for step, batch in enumerate(predict_data_loader):
            self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))
            if self.gpu_inference:
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.predict_step(batch)

    def save_inference_results(self, inputs, outputs):
        if self.options.model.name == "pixel2mesh":
            batch_size = inputs["images"].size(0)
            for i in range(batch_size):
                # basename, ext = os.path.splitext(inputs["filepath"][i])
                basename, ext = os.path.splitext(inputs["outpath"][i])
                mesh_center = np.mean(outputs["pred_coord_before_deform"][0][i].cpu().numpy(), 0)
                verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
                for k, vert in enumerate(verts):
                    meshname = basename + ".%d.obj" % (k + 1)
                    vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                    mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[k]))
                    # 确保目标文件夹存在
                    mesh_dir = os.path.dirname(meshname)
                    if not os.path.exists(mesh_dir):
                        os.makedirs(mesh_dir, exist_ok=True)
                        print(f"创建目录: {mesh_dir}")
                    np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")

def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Prediction Entrypoint')
    parser.add_argument('--options', 
                            help='experiment options file name', 
                            required=False, 
                            type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', 
                            default='C:/Users/Administrator/Desktop/Pixel2Mesh/checkpoints/3DBubbles_001000.pt',
                            # default='C:/Users/Administrator/Desktop/Pixel2Mesh/checkpoints/3DBubbles_001190.pt',
                            # default='C:/Users/Administrator/Desktop/Pixel2Mesh/checkpoints/3DBubbles_ST.pt',
                            # default='C:/Users/Administrator/Desktop/Pixel2Mesh/checkpoints/3DBubbles_ST1000.pt',
                            help='trained model file', 
                            type=str, 
                            required=False)
    parser.add_argument('--name', 
                            default='3DBubbles',
                            required=False, 
                            type=str)
    parser.add_argument('--folder', 
                            required=False, 
                            type=str)
    options.dataset.name += '_demo'
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 读取气泡裁剪文件夹路径
    # base_folder = "C:/codebase/BFRM/results/RB_bubble_flow1/2-analysis_results/bubble_crops/bubble_id"
    base_folder = "C:/codebase/BFRM/results/RB_bubble_flow2/2-analysis_results/bubble_crops/bubble_id"
    
    # 获取所有子文件夹
    bubble_folders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) 
                        if os.path.isdir(os.path.join(base_folder, d))]
    
    logger, writer = reset_options(options, args, phase='predict')
    
    # 处理每个气泡文件夹
    for folder in bubble_folders:
        print(f"正在处理文件夹: {folder}")
        options.dataset.predict.folder = folder
        
        # 为每个气泡创建一个预测器并执行预测
        predictor = Predictor(options, logger, writer)
        predictor.predict()


if __name__ == "__main__":
    main()
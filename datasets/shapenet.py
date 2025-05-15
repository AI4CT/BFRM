import json
import os
import pickle
import pyvista as pv

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Normalize

import config
from datasets.base_dataset import BaseDataset



class ShapeNet(BaseDataset):
    """Dataset wrapping images and target meshes for ShapeNet dataset.
    Arguments:
    """

    def __init__(self, root_dir, file_list_name, mesh_pos, normalization):

        self.root_dir = root_dir if file_list_name is None or file_list_name == "None" else os.path.join(root_dir, file_list_name)
        self.mesh_dir = os.path.join(self.root_dir, 'mesh')
        # self.images_dir = os.path.join(root_dir, 'images')
        # self.point_clouds_dir = os.path.join(root_dir, 'point_clouds')
        # self.normals_dir = os.path.join(root_dir, 'normals')
        # self.SH_complex_dir = os.path.join(root_dir, 'SH_complex')
        self.file_names = [f for f in os.listdir(self.mesh_dir) if f.endswith('.stl')][:1]

        with open(os.path.join(r"C:\Users\Administrator\Desktop\Pixel2Mesh\datasets\bubble.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}

        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.skip = 1

        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
    def __getitem__(self, index):

        mesh = pv.read(os.path.join(self.mesh_dir, self.file_names[index]))
        base_name = self.file_names[index].split('.stl')[0]
        # Load image
        # image_path = os.path.join(self.root_dir, f"projection/{base_name}/Sphere_-0.00_-1.00_0.00.png")
        image_path = os.path.join(self.root_dir, f"projection/{base_name}/Sphere_0.00_0.00_1.00.png")
        try:
            # 使用with语句自动关闭文件
            with Image.open(image_path) as image:
                # 一次性完成转换和归一化
                img = torch.from_numpy(
                    np.transpose(
                        np.array(image.convert('RGB'), dtype=np.float32) / 255.0,
                        (2, 0, 1)
                    )
                ).float()
            # 根据需要进行归一化
            img_normalized = self.normalize_img(img) if self.normalization else img
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")

        # load point cloud
        point_cloud = mesh.points
        point_cloud_trans = point_cloud - np.array(self.mesh_pos)
        # load normals 
        normals = mesh.point_normals
        for i in range(point_cloud.shape[0]):
            dot_product = np.dot(point_cloud[i] , normals[i])
            if dot_product < 0:
                normals[i] *= -1
        normals = torch.from_numpy(normals).float()
        point_cloud_trans = torch.from_numpy(point_cloud_trans).float()
        
        length = point_cloud_trans.shape[0]
        # for key, value in {
        #     "images": img_normalized,
        #     "images_orig": img,
        #     "points": point_cloud_trans,
        #     "normals": normals
        # }.items():
        #     if torch.isnan(value).any() or torch.isinf(value).any():
        #         raise ValueError(f"NaN or Inf detected in {key} for file {base_name}")
            
        return {
            "images": img_normalized,
            "images_orig": img,
            "points": point_cloud_trans,
            "normals": normals,
            "labels": self.labels_map['bubble'],
            "filename": base_name,
            "length": length
        }

    def __len__(self):
        return len(self.file_names)
    
# class ShapeNet(BaseDataset):
#     """
#     Dataset wrapping images and target meshes for ShapeNet dataset.
#     """

#     def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
#         super().__init__()
#         self.file_root = file_root
#         with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
#             self.labels_map = sorted(list(json.load(fp).keys()))
#         self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
#         # Read file list
#         with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
#             self.file_names = fp.read().split("\n")[:-1]
#         self.tensorflow = "_tf" in file_list_name # tensorflow version of data
#         self.normalization = normalization
#         self.mesh_pos = mesh_pos
#         self.resize_with_constant_border = shapenet_options.resize_with_constant_border

#     def __getitem__(self, index):
#         if self.tensorflow:
#             filename = self.file_names[index][17:]
#             label = filename.split("/", maxsplit=1)[0]
#             pkl_path = os.path.join(self.file_root, "data_tf", filename)
#             img_path = pkl_path[:-4] + ".png"
#             with open(pkl_path) as f:
#                 data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")
#             pts, normals = data[:, :3], data[:, 3:]
#             img = io.imread(img_path)
#             img[np.where(img[:, :, 3] == 0)] = 255
#             if self.resize_with_constant_border:
#                 img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
#                                        mode='constant', anti_aliasing=False)  # to match behavior of old versions
#             else:
#                 img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
#             img = img[:, :, :3].astype(np.float32)
#         else:
#             label, filename = self.file_names[index].split("_", maxsplit=1)
#             with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
#                 data = pickle.load(f, encoding="latin1")
#             img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

#         pts -= np.array(self.mesh_pos)
#         assert pts.shape[0] == normals.shape[0]
#         length = pts.shape[0]

#         img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
#         img_normalized = self.normalize_img(img) if self.normalization else img

#         return {
#             "images": img_normalized,
#             "images_orig": img,
#             "points": pts,
#             "normals": normals,
#             "labels": self.labels_map[label],
#             "filename": filename,
#             "length": length
#         }

#     def __len__(self):
#         return len(self.file_names)


class ShapeNetImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):
        super().__init__()
        self.normalization = normalization
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        self.glance = True
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img
        
        if self.glance:
            import matplotlib.pyplot as plt
            import os
            # 将张量转换为numpy数组并调整维度顺序
            img_np = img_normalized.cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            # 确保像素值在[0,1]范围内
            img_np = np.clip(img_np, 0, 1)
            # 获取文件名（不含路径）
            filename = os.path.basename(self.file_list[item])
            save_path = f"test_image/normalized_{filename}"
            plt.imsave(save_path, img_np)
            self.glance = False

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item],
            "outpath": self.file_list[item].replace(
                "C:/codebase/BFRM/results/yolo11l-obb/2-analysis_results/bubble_crops/bubble_id",
                "C:/codebase/BFRM/results/yolo11l-obb/4-reconstruction/bubble_obj"
            )
        }

    def __len__(self):
        return len(self.file_list)


def shapenet_collate(batch, num_points):
    if len(batch) > 1:
        all_equal = True
        for t in batch:
            if t["length"] != batch[0]["length"]:
                all_equal = False
                break
        points_orig, normals_orig = [], []
        if not all_equal:
            for t in batch:
                pts, normal = t["points"], t["normals"]
                length = pts.shape[0]
                choices = np.resize(np.random.permutation(length), num_points)
                t["points"], t["normals"] = pts[choices], normal[choices]
                # 检查并正确处理不同类型的数据
                if isinstance(pts, torch.Tensor):
                    points_orig.append(pts)
                else:
                    points_orig.append(torch.from_numpy(pts))
                    
                if isinstance(normal, torch.Tensor):
                    normals_orig.append(normal)
                else:
                    normals_orig.append(torch.from_numpy(normal))
            ret = default_collate(batch)
            ret["points_orig"] = points_orig
            ret["normals_orig"] = normals_orig
            return ret
    ret = default_collate(batch)
    ret["points_orig"] = ret["points"]
    ret["normals_orig"] = ret["normals"]
    return ret

# 避免使用 lambda 函数
class ShapenetCollator:
    def __init__(self, num_points):
        self.num_points = num_points
        
    def __call__(self, batch):
        return shapenet_collate(batch, self.num_points)

def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    return ShapenetCollator(num_points)
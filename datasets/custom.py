import os
from pathlib import Path

import gin
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from utils.frame_utils import read_gen


@gin.configurable()
class Custom(Dataset):
    def __init__(self, dataset_path, num_frames, min_dist_over_baseline=0.1, cam_format="TUM"):
        self.dataset_path = Path(dataset_path)
        self.image_formats = ["jpg", "jpeg", "png"]
        self.data_index = []
        for f in os.listdir(self.dataset_path / "images"):
            if os.path.splitext(f)[-1] in self.image_formats:
                self.image_formats = [os.path.splitext(f)[-1]]
                self.data_index.append(os.path.splitext(f)[0])
        self.data_index = sorted(self.data_index)
        if cam_format == "TUM":
            self.poses = np.zeros((len(self.data_index), 3, 4), dtype=np.float)
            cams_path = self.dataset_path / "cams.txt"
            poses = np.loadtxt(cams_path, dtype=np.float)[:, 1:]
            assert(len(poses) == len(self.data_index))
            for i in range(len(poses)):
                self.poses[i, :3, :3] = R.from_quat(poses[i, 3:]).as_matrix
                self.poses[i, :3, 3] = poses[i, :3]
            intrinsic_path = self.dataset_path / "intrinsic.txt"
            intrinsic = np.loadtxt(intrinsic_path, dtype=np.float)
            self.intrinsics = [intrinsic] * len(poses)
        if min_dist_over_baseline is not None:
            baselines = []
            for i in range(len(self.poses) - 1):
                baselines.append(np.linalg.norm(self.poses[i][:3, 3] - self.poses[i + 1][:3, 3]))
            self.min_dist = np.mean(baselines) * min_dist_over_baseline
        else:
            self.min_dist = None
        self.num_frames = num_frames
        assert(num_frames < len(self.data_index))
        self.image_format = self.image_formats[0]
        
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, index):
        start_index = min(max(0, index - self.neighbor_frames / 2), len(self.data_index) - 1 - self.neighbor_frames)
        indices = [index] + [i for i in range(start_index, start_index + self.neighbor_frames + 2) if i != index]
        images, poses, intrinsics = [], [], []
        
        for i in indices:
            image = read_gen(self.dataset_path / "images" / f"{self.data_index[i]}.{self.image_format}")
            images.append(image)
            poses.append(self.poses[i])
            intrinsics.append(self.intrinsics[i])

        if self.min_dist is None:
            cams_path0 = self.dataset_path / "cams" / f"{self.data_index[index]}_cam.txt"
            scale_info = np.loadtxt(cams_path0, skiprows=11, dtype=np.float)
            scale = 400 / scale_info[0]
        else:
            scale = 400 / self.min_dist

        images = np.stack(images, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)
        
        poses[:, :3, 3] *= scale

        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()
        
        return images, poses, intrinsics, [self.data_index[i] for i in indices], scale


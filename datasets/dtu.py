import glob
import os
from pathlib import Path

import gin
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import load_pair, random_scale_and_crop
from utils.frame_utils import read_gen

# for reproduce issue
training_set = [113, 14, 124, 111, 89, 45, 61, 104, 63, 22, 73, 39, 16, 42, 57, 8, 120, 119,
    83, 65, 103, 76, 87, 18, 58, 107, 91, 90, 99, 6, 41, 36, 46, 55, 109, 52, 101, 126, 25,
    19, 94, 88, 100, 7, 44, 122, 125, 51, 47, 96, 69, 98, 30, 68, 121, 127, 105, 93, 53, 102,
    64, 72, 27, 123, 128, 2, 116, 108, 20, 112, 92, 85, 50, 84, 70, 95, 26, 97, 60, 54, 31, 74, 71, 115]


val_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

test_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]



@gin.configurable()
class DTU(Dataset):
    def __init__(self, dataset_path="datasets/DTU", num_frames=10, light_number=-1,
        # by default use Y. Yao's pair selection
        pairs_provided=True,
        # other wise use our own criterion
        min_angle=3.0, max_angle=30.0
    ):
        self.dataset_path = Path(dataset_path)
        self.num_frames = num_frames
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.light_number = light_number
        self._build_dataset_index()
        self._load_poses()
        self.pairs_provided = pairs_provided
        if pairs_provided:
            self.pair_list = load_pair(self.dataset_path / "Cameras" / "pair.txt")


    def _theta_matrix(self, poses):
        delta_pose = np.matmul(poses[:,None], np.linalg.inv(poses[None,:]))
        dR = delta_pose[:, :, :3, :3]
        cos_theta = (np.trace(dR,axis1=2, axis2=3) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_theta))

    def _load_poses(self):
        pose_glob = glob.glob(self.dataset_path / "Cameras" / "*_cam.txt")
        extrinsics_list, intrinsics_list = [], []
        for cam_file in sorted(pose_glob):
            extrinsics = np.loadtxt(cam_file, skiprows=1, max_rows=4, dtype=np.float)
            intrinsics = np.loadtxt(cam_file, skiprows=7, max_rows=3, dtype=np.float)
            
            intrinsics[0] *= self.scale_between_image_depth
            intrinsics[1] *= self.scale_between_image_depth
            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)

        poses = np.stack(extrinsics_list, 0)
        intrinsics = np.stack(intrinsics_list, 0)

        # compute angle between all pairs of poses
        thetas = self._theta_matrix(poses)

        pose_graph = []
        for i in range(len(poses)):
            cond = (thetas[i]>self.min_angle) & (thetas[i]<self.max_angle)
            ixs, = np.where(cond)
            pose_graph.append(ixs)

        self.pose_graph = pose_graph
        self.poses = poses
        self.intrinsics = intrinsics
        self.theta_list = []
        for i in range(len(poses)):
            list_i = []
            for j in range(len(poses)):
                if thetas[i, j] > self.min_angle:
                    list_i.append((thetas[i, j], j))
            list_i = sorted(list_i)
            self.theta_list.append(list_i)

        
    def _build_dataset_index(self):
        self.dataset_index = []
        image_path = self.dataset_path / "Rectified"
        depth_path = self.dataset_path / "Depths"
        self.scale_between_image_depth = None
        self.scenes = {}
        for scene in [f"scan{i}" for i in training_set]:
            for k in range(7) if self.light_number == -1 else range(self.light_number, self.light_number + 1):
                images = sorted(glob.glob(image_path / scene / "*_%d_*.png" % k))
                depths = sorted(glob.glob(depth_path /  scene / "*.pfm"))
                if self.scale_between_image_depth is None:
                    self.scale_between_image_depth = int(read_gen(images[0]).shape[0] / read_gen(depths[0]).shape[0])
                scene_id = "%s_%d" % (scene, k)
                self.scenes[scene_id] = (images, depths)
                self.dataset_index += [(scene_id, i) for i in range(len(images))]


    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        # print(index, "index")
        scene_id, ref_id = self.dataset_index[index]
        image_list, depth_list = self.scenes[scene_id]

        if self.pairs_provided:
            neighbors = self.pair_list[ref_id]['pair'][:self.num_frames]
        else:
            if len(self.pose_graph[ref_id]) < self.num_frames:
                neighbors = np.random.choice([x[1] for x in self.theta_list[ref_id]][:self.num_frames * 2], self.num_frames, replace=False)
            else:
                neighbors = np.random.choice(self.pose_graph[ref_id], self.num_frames, replace=False)
            neighbors = neighbors.tolist()

        indicies = [ ref_id ] + neighbors
        images, depths, poses, intrinsics = [], [], [], []
        for i in indicies:
            image = read_gen(image_list[i])
            depth = read_gen(depth_list[i])
            pose = self.poses[i]
            calib = self.intrinsics[i]
            images.append(image)
            depths.append(depth)
            poses.append(pose)
            intrinsics.append(calib)

        images = np.stack(images, 0).astype(np.float32)
        depths = np.stack(depths, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)
        
        
        images = torch.from_numpy(images)
        depths = torch.from_numpy(depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()

        images, depths, intrinsics = random_scale_and_crop(images, depths, intrinsics)

        return images, depths, poses, intrinsics


@gin.configurable()
class DTUTest(Dataset):
    def __init__(self, dataset_path="datasets/DTU", scan=None, num_frames=None, subset=None, min_angle=4.0, max_angle=30.0, pairs_provided=True):
        self.dataset_path = dataset_path
        self.scan = scan
        self.num_frames = num_frames
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._build_dataset_index()
        self._load_poses()
        if pairs_provided:
            self.pair_list = load_pair(self.dataset_path / "Cameras" / "pair.txt")
        self.pairs_provided = pairs_provided
        if subset is None:
            self.dataset_index = list(range(len(self.image_list)))
        else:
            self.dataset_index = subset

    def _theta_matrix(self, poses):
        delta_pose = np.matmul(poses[:,None], np.linalg.inv(poses[None,:]))
        dR = delta_pose[:, :, :3, :3]
        cos_theta = (np.trace(dR,axis1=2, axis2=3) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_theta))

    def _load_poses(self):
        pose_glob = [self.dataset_path / "Cameras" / f"{i:08d}_cam.txt" for i in range(49)]
        extrinsics_list, intrinsics_list = [], []
        for cam_file in sorted(pose_glob):
            extrinsics = np.loadtxt(cam_file, skiprows=1, max_rows=4, dtype=np.float)
            intrinsics = np.loadtxt(cam_file, skiprows=7, max_rows=3, dtype=np.float)
            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)

        poses = np.stack(extrinsics_list, 0)
        intrinsics = np.stack(intrinsics_list, 0)

        # compute angle between all pairs of poses
        thetas = self._theta_matrix(poses)

        pose_graph = []
        for i in range(len(poses)):
            cond = (thetas[i]>self.min_angle) & (thetas[i]<self.max_angle)
            ixs, = np.where(cond)
            pose_graph.append(ixs)

        self.pose_graph = pose_graph
        self.poses = poses
        self.intrinsics = intrinsics
        
        self.theta_list = []

        for i in range(len(poses)):
            list_i = []
            for j in range(len(poses)):
                if thetas[i, j] > self.min_angle:
                    list_i.append((thetas[i, j], j))
            list_i = sorted(list_i)
            self.theta_list.append(list_i)

    def _build_dataset_index(self):
        image_glob = glob.glob(self.dataset_path / "Rectified" / self.scan / "rect_*_3_r5000.png")
        self.image_list = sorted(image_glob)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index0):
        # randomly sample neighboring frame
        ref_id = self.dataset_index[index0]
        if self.pairs_provided:
            if len(self.pair_list[ref_id]['pair']) >= self.num_frames:
                neighbors = self.pair_list[ref_id]['pair'][:self.num_frames]
            else:
                neighbors = [x for x in self.pair_list[ref_id]['pair']]
                head = 0
                goal = 0
                while len(neighbors) < self.num_frames:
                    if head < len(neighbors):
                        if len(self.pair_list[neighbors[head]]['pair']) > goal:
                            new_f = self.pair_list[neighbors[head]]['pair'][goal]
                        else:
                            break
                    else:
                        head = 0
                        goal += 1
                        continue
                    if not new_f in neighbors and new_f != ref_id:
                        neighbors.append(new_f)
                    head += 1
            indices = [ ref_id ] + neighbors

        else:
            if len(self.pose_graph[ref_id]) < self.num_frames:
                neighbors = np.random.choice([x[1] for x in self.theta_list[ref_id]][:self.num_frames * 2], self.num_frames, replace=False)
            else:
                neighbors = np.random.choice(self.pose_graph[ref_id], self.num_frames, replace=False)
            neighbors = neighbors.tolist()
            indices = [ ref_id ] + neighbors
        
        images, poses, intrinsics = [], [], []
        for i in indices:
            image = read_gen(self.image_list[i])
            pose = self.poses[i]
            calib = self.intrinsics[i].copy()
            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)

        images = np.stack(images, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)


        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()

        image_names = [str(i) for i in indices]

        return images, poses, intrinsics, image_names, 1.0




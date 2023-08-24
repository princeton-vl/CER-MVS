import os

import gin
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import load_pair
from utils.frame_utils import read_gen

training_set = ['Barn', 'Truck', 'Caterpillar', "Ignatius", 'Meetingroom', 'Church', 'Courthouse']
intermediate_set = ['Family','Francis','Horse','Lighthouse','M60','Panther','Playground','Train']
advanced_set = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]

@gin.configurable()
class TNT(Dataset):
    def __init__(self, dataset_path="datasets/TanksAndTemples", scan=None, num_frames=None, subset=None):
        self.scan = scan
        if scan in training_set:
            self.dataset_path = f"{dataset_path}/training_input/{scan}"
        elif scan in intermediate_set:
            self.dataset_path = f"{dataset_path}/tankandtemples/intermediate/{scan}"
        else:
            self.dataset_path = f"{dataset_path}/tankandtemples/advanced/{scan}"
        self.num_frames = num_frames
        self.pair_list = load_pair(os.path.join(self.dataset_path, 'pair.txt'))
        if subset is None:
            self.dataset_index = [i for i in range(len(self.pair_list['id_list']))]
        else:
            self.dataset_index = subset

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index0):
        index = self.dataset_index[index0]
        ref_id = self.pair_list['id_list'][index]
        if len(self.pair_list[ref_id]['pair']) >= self.num_frames:
            neighbors = [x for x in self.pair_list[ref_id]['pair'][:self.num_frames]]
        else:
            if self.pair_list[ref_id]['pair'] == []:
                min_ind = max(0, index - self.num_frames // 2)
                neighbors = [self.pair_list['id_list'][x] for x in list(range(min_ind, min_ind + self.num_frames + 1)) if x != index]
            else:
                neighbors = self.pair_list[ref_id]['pair']
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


        image_names = [f"{ref_id:08d}"] + [f"{x:08d}" for x in neighbors]
        images, poses, intrinsics = [], [], []
        for image_name in image_names:
            image = read_gen(os.path.join(self.dataset_path, "images", f"{image_name}.jpg"))
            cams_path = os.path.join(self.dataset_path, "cams", f"{image_name}_cam.txt")
            pose = np.loadtxt(cams_path, skiprows=1, max_rows=4, dtype=np.float)
            calib = np.loadtxt(cams_path, skiprows=7, max_rows=3, dtype=np.float)
            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)

        cams_path0 = os.path.join(self.dataset_path, "cams", f"{image_names[0]}_cam.txt")
        scale_info = np.loadtxt(cams_path0, skiprows=11, dtype=np.float)
        scale = 400 / scale_info[0]

        images = np.stack(images, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)

        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()
        
        return images, poses, intrinsics, image_names, scale


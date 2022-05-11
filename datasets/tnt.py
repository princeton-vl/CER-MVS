import numpy as np
import os
import gin
from utils.frame_utils import read_gen
from utils.data_utils import load_pair
import torch
from torch.utils.data import Dataset

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
        self.num_frames = num_frames + 1
        self.pair_list = load_pair(os.path.join(self.dataset_path, 'pair.txt'))
        self._build_dataset_index()
        self.subset = subset

    def _build_dataset_index(self):
        self.dataset_index = [i for i in range(len(self.pair_list['id_list']))]

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        if not self.subset is None and not index in self.subset: return []
        ix1 = self.dataset_index[index]
        ref_id = self.pair_list['id_list'][ix1]
        if len(self.pair_list[ref_id]['pair']) >= self.num_frames - 1:
            neighbors = [int(x) for x in self.pair_list[ref_id]['pair'][:self.num_frames-1]]
        else:
            if self.pair_list[ref_id]['pair'] == []:
                min_ind = max(0, ix1 - (self.num_frames - 1) // 2)
                neighbors = [int(self.pair_list['id_list'][x]) for x in list(range(min_ind, min_ind + self.num_frames)) if x != ix1]
            else:
                neighbors = [x for x in self.pair_list[ref_id]['pair']]
                head = 0
                goal = 0
                while len(neighbors) < self.num_frames - 1:
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
                neighbors = [int(x) for x in neighbors]


        indices = [ int(ref_id) ] + neighbors
        
        
        images, poses, intrinsics = [], [], []

        
        for i in indices:
            image = read_gen(os.path.join(self.dataset_path, "images", "%08d.jpg" % i))
            cams_path = os.path.join(self.dataset_path, "cams", "%08d_cam.txt" % i)
            pose = np.loadtxt(cams_path, skiprows=1, max_rows=4, dtype=np.float)
            calib = np.loadtxt(cams_path, skiprows=7, max_rows=3, dtype=np.float)
            images.append(image)
            poses.append(pose)
            intrinsics.append(calib)

        cams_path0 = os.path.join(self.dataset_path, "cams", "%08d_cam.txt" % indices[0])
        scale_info = np.loadtxt(cams_path0, skiprows=11, dtype=np.float)
        scale = 400 / scale_info[0]

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
        
        return images, poses, intrinsics, indices, scale


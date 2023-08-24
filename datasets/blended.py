import os

import gin
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.data_utils import load_pair, random_scale_and_crop
from utils.frame_utils import read_gen

training_set = ['5a3f4aba5889373fbbc5d3b5', '5bfc9d5aec61ca1dd69132a2', '5b908d3dc6ab78485f3d24a9', '5a0271884e62597cdee0d0eb', '5bb7a08aea1cfa39f1a947ab', '5be3a5fb8cfdd56947f6b67c', '5b69cc0cb44b61786eb959bf', '5ba75d79d76ffa2c86cf2f05', '5a69c47d0d5d0a7f3b2e9752', '5be883a4f98cee15019d5b83', '5a563183425d0f5186314855', '5aa515e613d42d091d29d300', '5adc6bd52430a05ecb2ffb85', '5bf26cbbd43923194854b270', '59f70ab1e5c5d366af29bf3e', '5c34529873a8df509ae57b58', '5abc2506b53b042ead637d86', '5bfd0f32ec61ca1dd69dc77b', '5a588a8193ac3d233f77fbca', '5ab85f1dac4291329b17cb50', '5b60fa0c764f146feef84df0', '5a57542f333d180827dfc132', '5a618c72784780334bc1972d', '5a6464143d809f1d8208c43c', '5bbb6eb2ea1cfa39f1af7e0c', '5ae2e9c5fe405c5076abc6b2', '5be47bf9b18881428d8fbc1d', '5b6eff8b67b396324c5b2672', '5b21e18c58e2823a67a10dd8', '5be4ab93870d330ff2dce134', '5aa0f9d7a9efce63548c69a1', '5bf03590d4392319481971dc', '5b4933abf2b5f44e95de482a', '5c34300a73a8df509add216d', '5bf18642c50e6f7f8bdbd492', '599aa591d5b41f366fed0d58', '59350ca084b7f26bf5ce6eb8', '5a969eea91dfc339a9a3ad2c', '5c1af2e2bee9a723c963d019', '59056e6760bb961de55f3501', '5bb8a49aea1cfa39f1aa7f75', '5aa235f64a17b335eeaf9609', '5bea87f4abd34c35e1860ab5', '5c062d84a96e33018ff6f0a6', '5b192eb2170cf166458ff886', '5947719bf1b45630bd096665', '5c1dbf200843bc542d8ef8c4', '5bd43b4ba6b28b1ee86b92dd', '5b2c67b5e0878c381608b8d8', '5bf21799d43923194842c001', '5a7d3db14989e929563eb153', '5af28cea59bc705737003253', '59f87d0bfa6280566fb38c9a', '58f7f7299f5b5647873cb110', '5bcf979a6d5f586b95c258cd', '5c1892f726173c3a09ea9aeb', '5b78e57afc8fcf6781d0c3ba', '5bff3c5cfe0ea555e6bcbf3a', '58c4bb4f4a69c55606122be4', '5a489fb1c7dab83a7d7b1070', '5af02e904c8216544b4ab5a2', '5bccd6beca24970bce448134', '5bfe5ae0fe0ea555e6a969ca', '5be3ae47f44e235bdbbc9771', '5a572fd9fc597b0478a81d14', '58d36897f387231e6c929903', '5ab8b8e029f5351f7f2ccf59', '5ab8713ba3799a1d138bd69a', '5a3cb4e4270f0e3f14d12f43', '5beb6e66abd34c35e18e66b9', '57f8d9bbe73f6760f10e916a', '58cf4771d0f5fb221defe6da', '5a4a38dad38c8a075495b5d2', '58eaf1513353456af3a1682a', '5b08286b2775267d5b0634ba', '5a48d4b2c7dab83a7d7b9851', '5c1f33f1d33e1f2e4aa6dda4', '5a3ca9cb270f0e3f14d0eddb', '5bf3a82cd439231948877aed', '5a8315f624b8e938486e0bd8', '5c20ca3a0843bc542d94e3e2', '59f363a8b45be22330016cad', '5afacb69ab00705d0cefdd5b', '5bf7d63575c26f32dbf7413b', '5b864d850d072a699b32f4ae', '5bc5f0e896b66a2cd8f9bd36', '5bce7ac9ca24970bce4934b6', '59e864b2a9e91f2c5529325f', '5a48c4e9c7dab83a7d7b5cc7', '5b62647143840965efc0dbde', '5947b62af1b45630bd0c2a02', '59e75a2ca9e91f2c5526005d', '5a48ba95c7dab83a7d7b44ed', '5acf8ca0f3d8a750097e4b15', '5a8aa0fab18050187cbe060e', '5b22269758e2823a67a3bd03', '5b6e716d67b396324c2d77cb', '5c2b3ed5e611832e8aed46bf', '5b3b353d8d46a939f93524b9', '5bf17c0fd439231948355385', '5c0d13b795da9479e12e2ee9', '59ecfd02e225f6492d20fcc9', '5c1b1500bee9a723c96c3e78', '5b271079e0878c3816dacca4', '59338e76772c3e6384afbb15', '5b558a928bbfb62204e77ba2']


subsets = ["dataset_full_res_0-29", "dataset_full_res_30-59", "dataset_full_res_60-89", "dataset_full_res_90-112"]

training_set = ['5bfe5ae0fe0ea555e6a969ca'] # !
subsets = ["dataset_full_res_0-29"] # !

@gin.configurable()
class Blended(Dataset):
    def __init__(self, dataset_path="datasets/BlendedMVS", num_frames=8, scaling="median"):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.scene_list = training_set
        self.scaling = scaling
        self.dataset_index = []
        for scene in tqdm(self.scene_list):
            flag = 0
            for subset in subsets:
                if scene in os.listdir(f"{dataset_path}/{subset}"):
                    flag = 1
                    break
            if flag == 0: continue
            pair_list = load_pair(os.path.join(dataset_path, subset, scene, scene, scene, 'cams', 'pair.txt'))
            for ref_id in pair_list['id_list']:
                if len(pair_list[ref_id]['pair']) < self.num_frames: continue
                self.dataset_index.append((scene, ref_id, pair_list[ref_id]['pair'][:self.num_frames]))

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        scene, ref_id, neib_ids = self.dataset_index[index]
        for subset in subsets:
            if scene in os.listdir(f"{self.dataset_path}/{subset}"):
                break
        
        indices = [ref_id] + neib_ids
        images, depths, poses, intrinsics = [], [], [], []

        for i in indices:
            image_path = os.path.join(self.dataset_path, subset, scene, scene, scene, "blended_images", "%08d.jpg" % i)
            depth_path = os.path.join(self.dataset_path, subset, scene, scene, scene, "rendered_depth_maps", "%08d.pfm" % i)
            try:
                image = read_gen(image_path)
                depth = read_gen(depth_path)
            except:
                print("data incomplete", scene, image_path, depth_path)
                assert(0)
            cams_path = os.path.join(self.dataset_path, subset, scene, scene, scene, "cams", "%08d_cam.txt" % i)
            pose = np.loadtxt(cams_path, skiprows=1, max_rows=4, dtype=np.float)
            calib = np.loadtxt(cams_path, skiprows=7, max_rows=3, dtype=np.float)
            
            images.append(image)
            depths.append(depth)
            poses.append(pose)
            intrinsics.append(calib)

        images = np.stack(images, 0).astype(np.float32)
        depths = np.stack(depths, 0).astype(np.float32)
        poses = np.stack(poses, 0).astype(np.float32)
        intrinsics = np.stack(intrinsics, 0).astype(np.float32)        

        if self.scaling == "median":
            depth_f = depths.reshape((-1,))
            scale = 600 / np.median(depth_f[depth_f > 0])
        else:
            cams_path0 = os.path.join(self.dataset_path, scene, scene, scene, "cams", "%08d_cam.txt" % indices[0])
            scale_info = np.loadtxt(cams_path0, skiprows=11, dtype=np.float)
            scale = 400 / scale_info[0]

        depths *= scale
        poses[:, :3, 3] *= scale

        images = torch.from_numpy(images)
        depths = torch.from_numpy(depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        # channels first
        images = images.permute(0, 3, 1, 2)
        images = images.contiguous()

        images, depths, intrinsics = random_scale_and_crop(images, depths, intrinsics)
        return images, depths, poses, intrinsics
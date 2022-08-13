import gin
import argparse
import os
import time
import numpy as np
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from core.raft import RAFT
from datasets.dtu import DTU, DTUTest
from datasets.blended import Blended
from datasets.tnt import TNT
from utils.frame_utils import write_pfm
from utils.data_utils import scale_operation, crop_operation


dataset_dict = {
    "DTU": DTU,
    "DTUTest": DTUTest,
    "Blended": Blended,
    "TNT": TNT
}


@gin.configurable()
def inference(
        datasetname,
        scan,
        ckpt,
        output_folder,
        rescale=1,
        num_frame=10,
        crop=None,
        subset=None,
        do_report=False,
    ):

    if not subset is None:
        start, end, step = subset
        subset = list(range(start, end, step))
    model = RAFT().cuda()

    if ckpt is not None:
        tmp = torch.load(ckpt)
        if list(tmp.keys())[0][:7] == "module.":
            model = nn.DataParallel(model)
        model.load_state_dict(tmp, strict=True)
    model.eval()
    
    gpuargs = {'num_workers': 4, 'drop_last' : False, 'shuffle': False, 'pin_memory': True}

    test_dataset = dataset_dict[datasetname](
        scan=scan,
        num_frames=num_frame,
        subset=subset
    )
    test_loader = DataLoader(test_dataset, batch_size=1, **gpuargs)

    subprocess.call("mkdir -p %s" % os.path.join(output_folder, scan, "depths"), shell=True)

    with torch.no_grad():
        for i, data_blob in enumerate(test_loader):
            print(i)
            if not subset is None and not i in subset: continue
            images, poses, intrinsics, indices, scale = data_blob
            poses = poses.cuda()
            images = images.squeeze(0)
            intrinsics = intrinsics.squeeze(0)
            images, intrinsics = scale_operation(images, intrinsics, rescale)
            if not crop is None:
                crop_h, crop_w = crop
                images, intrinsics = crop_operation(images, intrinsics, crop_h, crop_w)
            images = images.unsqueeze(0).cuda()
            intrinsics = intrinsics.unsqueeze(0).cuda()
            if do_report:
                tic = time.time()
            disp_est = model(images, poses, intrinsics, do_report=do_report)[-1] * scale.cuda()
            if do_report:
                print(f"per view time: {time.time() - tic}")
            res = disp_est.cpu().numpy()[0, 0]
            im = np.where(res == 0, 0, 1 / res).astype(np.float32)
            write_pfm(os.path.join(f"{output_folder}/{scan}/depths/%08d_scale{rescale}_nf{num_frame}.pfm" % indices[0]), im)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gin_config', nargs='+', default=['inference_DTU'],
                        help='Set of config files for gin (separated by spaces) '
                        'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--gin_param', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                        'e.g. --gin_param module_1.a=2 module_2.b=3')
    args = parser.parse_args()
    gin_files = [f'configs/{g}.gin' for g in args.gin_config]
    gin.parse_config_files_and_bindings(
        gin_files, args.gin_param, skip_unknown=True)

    inference()
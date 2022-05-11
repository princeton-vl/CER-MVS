import numpy as np
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import gin
from utils.frame_utils import readPFM
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import math
from utils.bilinear_sampler import bilinear_sampler
from datasets.tnt import training_set as tnt_training_set
from datasets.tnt import intermediate_set as tnt_intermediate_set
from datasets.tnt import advanced_set as tnt_advanced_set
cudnn.benchmark = True


# Modified Pytorch Version of D2HC-RMVSNet [J. Yan et al., 2020]


# read intrinsics and extrinsics
def read_camera_parameters(filename,scale,index,flag):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    intrinsics[:2, :] *= scale

    if (flag==0):
        intrinsics[0,2]-=index
    else:
        intrinsics[1,2]-=index
  
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data



# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    batch, height, width = depth_ref.shape
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1=4.4, thre2=1430.):
    batch, height, width = depth_ref.shape
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    masks=[]
    for i in range(2,11):
        mask = torch.logical_and(dist < i/thre1, relative_depth_diff < i/thre2)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff


@gin.configurable()
def fusion(
        dataset, scan,
        output_folder,
        suffix="",
        glb=0.25,
        downsample=1,
        rescale=1
    ):

    output_scan_folder = os.path.join(output_folder, scan)
    # the pair file
    if dataset == "DTU":
        image_folder = os.path.join("datasets/DTU/Rectified", scan)
        cams_folder = "datasets/DTU/Cameras"
        pair_file = "datasets/DTU/Cameras/pair.txt"
    elif dataset == "TNT":
        if scan in tnt_training_set:
            scan_folder = os.path.join("datasets/TanksAndTemples/training_input", scan)
        elif scan in tnt_intermediate_set:
            scan_folder = os.path.join("datasets/TanksAndTemples/tankandtemples/intermediate", scan)
        elif scan in tnt_advanced_set:
            scan_folder = os.path.join("datasets/TanksAndTemples/tankandtemples/advanced", scan)
        image_folder = os.path.join(scan_folder, "images")
        cams_folder = os.path.join(scan_folder, "cams")
        pair_file = os.path.join(scan_folder, "pair.txt")

    
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)[::downsample]
    n_view = len(pair_data)

    # for each reference view and the corresponding source views
    ct2 = -1


    all_images = None
    all_depths = None
    all_intrinsics = None
    all_extrinsics = None

    if dataset == "DTU":
        n_images = 49
    elif dataset == "TNT":
        indices = [int(x[:-4]) for x in os.listdir(os.path.join(scan_folder, 'images'))]
        n_images = np.max(indices) + 1


    for i in range(n_images):
        print(f"loading {i}")
        try:
            if dataset == "DTU":
                path = f'{image_folder}/rect_{i + 1:03d}_3_r5000.png'
                ref_img = read_img(path)
            elif dataset == "TNT":
                ref_img = read_img(f'{image_folder}/{i:08d}.jpg')
        except:
            print("no image, skip")
            continue
        ref_depth_est = readPFM(os.path.join(output_scan_folder, f'depths/{i:08d}{suffix}.pfm'))
        h, w = ref_depth_est.shape
        ref_depth_est = cv2.resize(ref_depth_est, (int(w * rescale), int(h * rescale)))
        scale=float(ref_depth_est.shape[0])/ref_img.shape[0]

        index=int((int(ref_img.shape[1]*scale)-ref_depth_est.shape[1])/2)
        flag=0
        if (ref_depth_est.shape[1]/ref_img.shape[1]>scale):
            scale=float(ref_depth_est.shape[1])/ref_img.shape[1]
            index=int((int(ref_img.shape[0]*scale)-ref_depth_est.shape[0])/2)
            flag=1
        ref_img = cv2.resize(ref_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if flag == 0:
            index = int(math.ceil((ref_img.shape[1] - ref_depth_est.shape[1]) / 2))
        else:
            index = int(math.ceil((ref_img.shape[0] - ref_depth_est.shape[0]) / 2))
        if (flag==0):
            ref_img=ref_img[:,index:ref_depth_est.shape[1] + index,:]
        else:
            ref_img=ref_img[index:ref_img.shape[0]-index,:,:]

        ref_intrinsics, ref_extrinsics = read_camera_parameters(f'{cams_folder}/{i:08d}_cam.txt',scale,index,flag)

        if i == 0:
            all_images = np.zeros((n_images, *ref_img.shape))
            all_depths = np.zeros((n_images, *ref_depth_est.shape))
            all_extrinsics = np.zeros((n_images, *ref_extrinsics.shape))
            all_intrinsics = np.zeros((n_images, *ref_intrinsics.shape))
            init_h_image = ref_img.shape[0]
            init_w_image = ref_img.shape[1]
            init_h_depth = ref_depth_est.shape[0]
            init_w_depth = ref_depth_est.shape[1]

        if ref_img.shape == all_images.shape[1:]:
            all_images[i] = ref_img
        else:
            small_h = min(ref_img.shape[0], init_h_image)
            small_w = min(ref_img.shape[1], init_w_image)
            all_images[i, :small_h, :small_w] = ref_img[:small_h, :small_w]
        if ref_depth_est.shape == all_depths.shape[1:]:
            all_depths[i] = ref_depth_est
        else:
            small_h = min(ref_depth_est.shape[0], init_h_depth)
            small_w = min(ref_depth_est.shape[1], init_w_depth)
            all_depths[i, :small_h, :small_w] = ref_depth_est[:small_h, :small_w]

        all_extrinsics[i] = ref_extrinsics
        all_intrinsics[i] = ref_intrinsics



    h, w = all_depths[0].shape
    all_images = torch.from_numpy(np.stack(all_images)).cuda()
    all_depths = torch.from_numpy(np.stack(all_depths)).float().cuda()
    all_intrinsics = torch.from_numpy(np.stack(all_intrinsics)).float().cuda()
    all_extrinsics = torch.from_numpy(np.stack(all_extrinsics)).float().cuda()


    thre_left = -2
    thre_right = 2

    tot_iter = 10
    for iter in range(tot_iter):
        thre = (thre_left + thre_right) / 2
        print(f"{iter} {10 ** thre}")


        depth_est = torch.zeros((n_view, h, w)).cuda()
        # thre = 1.75
        geo_mask_all = []
        ref_id = 0

        for ref_view, src_views in pair_data:
            print(f"ref view {ref_view}")
            ct2 += 1

            # load the reference image
            ref_img = all_images[ref_view]
            # load the estimated depth of the reference view
            ref_depth_est = all_depths[ref_view]

            # load the camera parameters
            ref_extrinsics = all_extrinsics[ref_view]
            ref_intrinsics = all_intrinsics[ref_view]

            # compute the geometric mask


            n = 1 + len(src_views)

            print(src_views)
            src_intrinsics, src_extrinsics = all_intrinsics[src_views], all_extrinsics[src_views]
            src_depth_est = all_depths[src_views]
            n_src = len(src_views)
            ref_depth_est = ref_depth_est.unsqueeze(0).repeat(n_src, 1, 1)
            ref_intrinsics = ref_intrinsics.unsqueeze(0).repeat(n_src, 1, 1)
            ref_extrinsics = ref_extrinsics.unsqueeze(0).repeat(n_src, 1, 1)
            if n_src == 0:
                ref_id += 1
                continue
            masks, geo_mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(ref_depth_est, ref_intrinsics, # parallelize it!
                                                                                        ref_extrinsics,
                                                                                        src_depth_est,
                                                                                        src_intrinsics, src_extrinsics,
                                                                                        10 ** thre * 4, 10 ** thre * 1300)

            geo_mask_sums=[]
            for i in range(2,n):
                geo_mask_sums.append(masks[i-2].sum(dim=0).int()) #masks[i-2][0].int())
            
            geo_mask_sum = geo_mask.sum(dim=0)

            geo_mask=geo_mask_sum>=n

            for i in range (2, n):
                geo_mask=torch.logical_or(geo_mask,geo_mask_sums[i-2]>=i)
            depth_est[ref_id] = (depth_reprojected.sum(dim=0) + ref_depth_est[0]) / (geo_mask_sum + 1)

            del masks

            geo_mask_all.append(geo_mask.float().mean().item())
            torch.cuda.empty_cache()


            if iter == tot_iter - 1:
                ref_intrinsics = ref_intrinsics[0]
                ref_extrinsics = ref_extrinsics[0]
                
                os.makedirs(os.path.join(output_scan_folder, "mask"), exist_ok=True)

                depth_est_averaged = depth_est[ref_id].cpu().numpy()
                geo_mask = geo_mask.cpu().numpy()


                save_mask(os.path.join(output_scan_folder, f"mask/{ref_view:0>8}{suffix}.png"), geo_mask)
                print(f"processing {scan}, ref-view{ref_view}, mask:{geo_mask.mean()}")
                valid_points = geo_mask
            
                ref_img = ref_img.cpu().numpy()

                

                height, width = depth_est_averaged.shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
                
                # print("valid_points", valid_points.mean())
                x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
                color = ref_img[:, :, :][valid_points]  # hardcoded for DTU dataset
                xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics.cpu().numpy()),
                                    np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = np.matmul(np.linalg.inv(ref_extrinsics.cpu().numpy()),
                                    np.vstack((xyz_ref, np.ones_like(x))))[:3]
                vertexs.append(xyz_world.transpose((1, 0)))
                vertex_colors.append((color * 255).astype(np.uint8))
            ref_id += 1
        if np.mean(geo_mask_all) >= glb:
            thre_left = thre
        else:
            thre_right = thre



    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')

    plyfilename = os.path.join(output_folder, scan + '.ply')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')
    parser.add_argument('-g', '--gin_config', nargs='+', default=[],
                        help='Set of config files for gin (separated by spaces) '
                        'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--gin_param', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                        'e.g. --gin_param module_1.a=2 module_2.b=3')

    args = parser.parse_args()

    gin_files = [f'configs/{g}.gin' for g in args.gin_config]
    gin.parse_config_files_and_bindings(
        gin_files, args.gin_param, skip_unknown=True)


    fusion()

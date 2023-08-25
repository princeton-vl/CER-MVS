import argparse
from pathlib import Path

import gin

from datasets import get_test_data_loader
from fusion import fusion
from inference import inference
from multires import multires

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    output_folder = Path("results")

    # custom
    custom_dataset_path = "datasets/custom"
    custom_output_folder = output_folder / "custom"

    # use i = 0 pass to get min depth estimation
    for i, (rescale, num_frames) in enumerate([(0.5, 10), (1, 15), (2, 25)]):
        if i == 0: args = {}
        else: args = {"min_dist_over_baseline": None}
        data_loader = get_test_data_loader("Custom", dataset_path=custom_dataset_path, num_frames=num_frames, **args)
        inference(
            data_loader,
            ckpt="pretrained/train_BlendedMVS.pth",
            output_folder=custom_output_folder,
            rescale=rescale,
            do_report=True,
            write_min_depth=("datasets/custom/min_depth" if i == 0 else None),
        )
    # Multi Res Fusion
    multires(custom_output_folder, suffix1="_nf15", suffix2="_nf25", visualize=True)
    # Adaptive Threshold Fusion
    data_loader = get_test_data_loader("Custom", dataset_path=custom_dataset_path, num_frames=10)
    fusion(data_loader, custom_output_folder, rescale=1, suffix="_nf15_nf25_th0.02")

    # # OR if you don't want miltiresoultion step
    # for i, (rescale, num_frames) in enumerate([(0.5, 10), (1, 15)]):
    #     if i == 0: args = {}
    #     else: args = {"min_dist_over_baseline": None}
    #     data_loader = get_test_data_loader("Custom", dataset_path=custom_dataset_path, num_frames=num_frames, **args)
    #     inference(
    #         data_loader,
    #         ckpt="pretrained/train_BlendedMVS.pth",
    #         output_folder=custom_output_folder,
    #         rescale=rescale,
    #         do_report=True,
    #         write_min_depth=("datasets/custom/min_depth" if i == 0 else None),
    #     )
    # data_loader = get_test_data_loader("Custom", dataset_path=custom_dataset_path, num_frames=10)
    # fusion(data_loader, custom_output_folder, rescale=1, suffix="_scale1_nf15")

    # OR if you only want to run it with only a subset of the dataset, run python demo_custom.py -p 'get_test_data_loader.subset=(start,end,step)'
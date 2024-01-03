import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from loguru import logger

from qct_utils.ctscan.ct_loader import CTAnnotLoader
from qct_utils.schema.ct.chestct import ChestCTMaster

# ANSI codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def combine_scans(folder_path):
    train_csv_files = [f"folds_{i}.csv" for i in range(4)]

    # Initialize lists to store the 'scans' data
    train_scans, val_scans = [], []

    for csv_file in train_csv_files:
        file_path = os.path.join(folder_path, csv_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            train_scans.append(df["scans"])
        else:
            print(f"File not found: {file_path}")
    train_scans_list = pd.concat(train_scans, ignore_index=True).tolist()

    # Read and combine 'scans' data from folds_4.csv for validation
    val_file_path = os.path.join(folder_path, "folds_4.csv")
    if os.path.exists(val_file_path):
        val_df = pd.read_csv(val_file_path)
        val_scans = val_df["scans"].tolist()
    else:
        print(f"Validation file not found: {val_file_path}")

    return train_scans_list, val_scans


def normalize_image(
    img,
    a_min: float,
    a_max: float,
    b_min: float = None,
    b_max: float = None,
    clip: bool = True,
    renorm: bool = False,
):
    if a_max - a_min == 0.0:
        print("Divide by zero (a_min == a_max)")
        if b_min is None:
            return img - a_min
        return img - a_min + b_min
    img = (img - a_min) / (a_max - a_min)
    if (b_min is not None) and (b_max is not None):
        img = img * (b_max - b_min) + b_min
    if clip:
        img = np.clip(img, b_min, b_max)
    if renorm:
        img = 2 * img - 1  # renormalize the image from [-1, 1]
    return img.astype(np.float32)


def get_norm_img_and_combined_mask_for_scan(
    chestct: ChestCTMaster, return_sitk: bool = False
):
    img = chestct.scan.array
    img = normalize_image(
        img=img, a_min=-1024.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True
    )
    combined_mask = np.zeros_like(img, dtype=np.uint8)

    for nod in chestct.nodules:
        mask = nod.gt.annot.mask.get_numpy_array().astype(np.uint8)
        combined_mask += mask

    # Threshold the combined mask to ensure it contains only 0s and 1s
    combined_mask = np.where(combined_mask > 1, 1, combined_mask)

    if return_sitk:
        return sitk.GetImageFromArray(img), sitk.GetImageFromArray(combined_mask)

    return img, combined_mask


if __name__ == "__main__":
    MODE = "train"  # train or val
    json_path = "/home/users/utkarsh.singh/qct/qct_data/studies/segm_train_data_distribution/segm_test_data.json"

    num_readers = 2
    dataset_name = "lidc"
    save_cache_root = f"/cache/fast_data_nas8/utkarsh/segm_cache/sam/{dataset_name}"
    csv_loc = f"/home/users/utkarsh.singh/qct/medsam/SAM-Med3D/data/csvs/{dataset_name}_majority_{num_readers}reader.csv"
    # scans_root = f"/home/users/utkarsh.singh/qct/medsam/SAM-Med3D/data/pt/{dataset_name}.pt"
    scans_root = f"/cache/fast_data_nas72/qct/data_governance/series_dicts/{dataset_name}.pt"
    os.makedirs(save_cache_root, exist_ok=True)

    logger.info(f"{RED}Generating {MODE} cache for SAM-Med3D training for {dataset_name}{RESET}")

    # lidc_folds_folder = "/home/users/utkarsh.singh/qct/qct_nodule_detection/studies/only_lidc/folds3"
    # nlst_folds_folder = "/cache/fast_data_nas8/cache_3d/qct_det_cache/nlst/folds"

    # train_scans, val_scans = combine_scans(folder_path=nlst_folds_folder)

    # scan_names = train_scans if MODE == "train" else val_scans  # train/val scans sid
    # scan_names = [line.strip() for line in open('test_sid.txt', 'r')] # test scans sid

    # imgs_dir = (
    #     f"{save_cache_root}/imagesTr"
    #     if MODE == "train"
    #     else f"{save_cache_root}/imagesVal"
    # )
    # labels_dir = (
    #     f"{save_cache_root}/labelsTr"
    #     if MODE == "train"
    #     else f"{save_cache_root}/labelsVal"
    # )
    
    imgs_dir = f"{save_cache_root}/imagesTs"
    labels_dir = f"{save_cache_root}/labelsTs"

    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    ds = CTAnnotLoader(scans_root=scans_root, csv_loc=csv_loc) # for LIDC; used this so far: "/cache/fast_data_nas6/chestct/lidc_nii/scans_cct", for NLST: "/cache/fast_data_nas6/chestct/nlst/lung_cancer_pos_nii/bucket_0_1_2/scans_cct"

    with open(json_path) as f:
        j = json.load(f)
    nids_to_take = list(j[dataset_name].keys()) # for nlst 2 reader majority: len(nids_to_take) == 2260
    
    df = pd.read_csv(csv_loc) # for nlst 2 reader majority: df.shape == (1162, 3)
    nodule_identifier_column = 'annot_id' if 'annot_id' in df.columns else 'Nodule_name'
    df_ = df[df[nodule_identifier_column].isin(nids_to_take)]
    nids_to_take_final = df_[nodule_identifier_column].values
    scan_names = list(set([n.split('_')[0] for n in nids_to_take_final]))

    for scan_name in tqdm(scan_names):
        logger.info(f"{CYAN}Processing:{RESET} {scan_name}")
        tt = ds[scan_name]
        img, gt = get_norm_img_and_combined_mask_for_scan(chestct=tt, return_sitk=True)

        sitk.WriteImage(img, f"{imgs_dir}/{scan_name}.nii.gz")
        sitk.WriteImage(gt, f"{labels_dir}/{scan_name}.nii.gz")

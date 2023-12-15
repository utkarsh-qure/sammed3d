import os
import json
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchio as tio

from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL
from utils.metrics import compute_dice, compute_iou

from segment_anything.build_sam3D import sam_model_registry3D

parser = argparse.ArgumentParser()
parser.add_argument(
    "-tdp",
    "--test_data_path",
    type=str,
    default="/cache/fast_data_nas8/utkarsh/sam_med3d_cache",
)
parser.add_argument(
    "-vp", "--vis_path", type=str, default="./results/visualization"
)  # ./results/visualization2d for medsam results
parser.add_argument(
    "-cp",
    "--checkpoint_path",
    type=str,
    default="./work_dir/lidc_finetune/sam_model_dice_best.pth",
)  # ./ckpt/medsam2d_point_prompt.pth for medsam

parser.add_argument("--image_size", type=int, default=256)  # 1024 for medsam
parser.add_argument("--crop_size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("-mt", "--model_type", type=str, default="vit_b_ori")
parser.add_argument("-nc", "--num_clicks", type=int, default=10)
parser.add_argument("-pm", "--point_method", type=str, default="random")
parser.add_argument("-dt", "--data_type", type=str, default="Ts")  # Tr, Val, Ts

parser.add_argument("--threshold", type=int, default=0)
parser.add_argument("--dim", type=int, default=3)  # 2 for medsam
parser.add_argument("--split_idx", type=int, default=0)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--ft2d", action="store_true", default=False)  # set for medsam
parser.add_argument("--seed", type=int, default=2023)

args = parser.parse_args()

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    "random": get_next_click3D_torch_2,
}


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    if args.ft2d and ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        pad = None
    return masks, pad


def sam_decoder_inference(
    target_size,
    points_coords,
    points_labels,
    model,
    image_embeddings,
    mask_inputs=None,
    multimask=False,
):
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points_coords.to(model.device), points_labels.to(model.device)),
            boxes=None,
            masks=mask_inputs,
        )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )

    low_res_masks = torch.sigmoid(low_res_masks)  # convert to probabilities
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i : i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(
        low_res_masks,
        (target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return masks, low_res_masks, iou_predictions


def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.0
    return new_arr


def random_point_sampling(mask, get_point=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor(
            [label], dtype=torch.int
        )
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = (
            torch.as_tensor(coords[indices], dtype=torch.float),
            torch.as_tensor(labels[indices], dtype=torch.int),
        )
        return coords, labels


def finetune_model_predict3D(
    img3D,
    gt3D,
    sam_model_tune,
    device="cuda",
    click_method="random",
    num_clicks=10,
    prev_masks=None,
):
    # img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    # img3D = img3D.unsqueeze(dim=1)
    click_points = []
    click_labels = []

    pred_list = []

    iou_list = []
    dice_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(
        prev_masks.float(),
        size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4),
    )

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(
            img3D.to(device)
        )  # (1, 384, 16, 16, 16)
    for num_click in range(num_clicks):
        with torch.no_grad():
            if num_click > 1:
                click_method = "random"
            batch_points, batch_labels = click_methods[click_method](
                prev_masks.to(device), gt3D.to(device)
            )

            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
                multimask_output=False,
            )
            prev_masks = F.interpolate(
                low_res_masks,
                size=gt3D.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(
                np.uint8
            )  # confidence threshold 0.5
            pred_list.append(medsam_seg)

            iou_list.append(
                round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4)
            )
            dice_list.append(
                round(
                    compute_dice(
                        gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg
                    ),
                    4,
                )
            )

    return pred_list, click_points, click_labels, iou_list, dice_list


if __name__ == "__main__":
    # all_dataset_paths = glob(os.path.join(args.test_data_path, "*", "*"))
    all_dataset_paths = glob(args.test_data_path)
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(
            mask_name="label",
            target_shape=(args.crop_size, args.crop_size, args.crop_size),
        ),
    ]

    test_dataset = Dataset_Union_ALL(
        paths=all_dataset_paths,
        mode="Ts",
        data_type=args.data_type,
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, sampler=None, batch_size=1, shuffle=True
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)

    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict["model_state_dict"]
        sam_model_tune.load_state_dict(state_dict)

    all_iou_list = []
    all_dice_list = []

    out_dice = dict()
    out_dice_all = OrderedDict()

    dataset = "lidc_test" if args.data_type == "Ts" else "lidc_val"
    vis_root = os.path.join(os.path.dirname(__file__), args.vis_path, dataset)

    for batch_data in tqdm(test_dataloader):
        image3D, gt3D, img_name = batch_data
        sz = image3D.size()
        if sz[2] < args.crop_size or sz[3] < args.crop_size or sz[4] < args.crop_size:
            print("[ERROR] wrong size", sz, "for", img_name)

        pred_path = os.path.join(
            vis_root,
            os.path.basename(img_name[0]).replace(
                ".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"
            ),
        )

        if os.path.exists(pred_path):
            iou_list, dice_list = [], []
            for iter in range(args.num_clicks):
                curr_pred_path = os.path.join(
                    vis_root,
                    os.path.basename(img_name[0]).replace(
                        ".nii.gz", f"_pred{iter}.nii.gz"
                    ),
                )
                medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                iou_list.append(
                    round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4)
                )
                dice_list.append(
                    round(
                        compute_dice(
                            gt3D[0][0].detach().cpu().numpy().astype(np.uint8),
                            medsam_seg,
                        ),
                        4,
                    )
                )
        else:
            # norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            (
                seg_mask_list,
                points,
                labels,
                iou_list,
                dice_list,
            ) = finetune_model_predict3D(
                image3D,
                gt3D,
                sam_model_tune,
                device=device,
                click_method=args.point_method,
                num_clicks=args.num_clicks,
                prev_masks=None,
            )

            os.makedirs(vis_root, exist_ok=True)
            points = [point_.cpu().numpy() for point_ in points]
            labels = [label_.cpu().numpy() for label_ in labels]
            pt_info = dict(points=points, labels=labels)
            print(
                "save to",
                os.path.join(
                    vis_root,
                    os.path.basename(img_name[0]).replace(".nii.gz", "_pred.nii.gz"),
                ),
            )
            pt_path = os.path.join(
                vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pt.pkl")
            )
            pickle.dump(pt_info, open(pt_path, "wb"))
            for idx, pred3D in enumerate(seg_mask_list):
                out = sitk.GetImageFromArray(pred3D)
                sitk.WriteImage(
                    out,
                    os.path.join(
                        vis_root,
                        os.path.basename(img_name[0]).replace(
                            ".nii.gz", f"_pred{idx}.nii.gz"
                        ),
                    ),
                )

        per_iou = max(iou_list)
        all_iou_list.append(per_iou)
        all_dice_list.append(max(dice_list))
        print(dice_list)
        out_dice[img_name] = max(dice_list)
        cur_dice_dict = OrderedDict()
        for i, dice in enumerate(dice_list):
            cur_dice_dict[f"{i}"] = dice
        out_dice_all[img_name[0]] = cur_dice_dict

    print("Mean IoU : ", sum(all_iou_list) / len(all_iou_list))
    print("Mean Dice: ", sum(all_dice_list) / len(all_dice_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split("/")[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split("/")[-4]
        final_dice_dict[organ][k] = v

    save_name = f"{vis_root}/out_dice.py"
    if args.split_num > 1:
        save_name = save_name.replace(".py", f"_s{args.split_num}i{args.split_idx}.py")

    print("Save to", save_name)
    with open(save_name, "w") as f:
        f.writelines(f"# mean dice: \t{np.mean(all_dice_list)}\n")
        f.writelines("dice_Ts = {")
        for k, v in out_dice.items():
            f.writelines(f"'{str(k[0])}': {v},\n")
        f.writelines("}")

    with open(save_name.replace(".py", ".json"), "w") as f:
        json.dump(final_dice_dict, f, indent=4)

    print("Done")

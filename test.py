import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from collections import OrderedDict

import torch
import torch.nn.functional as F

from build import build_model, get_test_dataloader
from utils.metrics import compute_dice, compute_iou
from utils.click_method import get_next_click3D_torch_2


parser = argparse.ArgumentParser()
parser.add_argument(
    "-tdp",
    "--test_data_path",
    type=str,
    default="/cache/fast_data_nas8/utkarsh/segm_cache/sam",
)
parser.add_argument(
    "-vp", "--vis_path", type=str, default="./results/dsb_union_2reader_ft_turbo"
)  # ./results/medsam2d for medsam results
parser.add_argument(
    "-cp",
    "--checkpoint_path",
    type=str,
    default="/cache/fast_data_nas8/utkarsh/training/sammed3d_workdir/dsb_union_2reader_ft_turbo/sam_model_dice_best.pth",
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


def get_points(prev_masks, gt3D, click_idx, click_method, device):
    if click_idx > 1:
        click_method = "random" # first mask has to be from gt
    batch_points, batch_labels = click_methods[click_method](
        prev_masks.to(device), gt3D.to(device)
    )

    points_co = torch.cat(batch_points, dim=0).to(device)
    points_la = torch.cat(batch_labels, dim=0).to(device)

    return points_co, points_la


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
    for click_idx in range(num_clicks):
        with torch.no_grad():
            points_input, labels_input = get_points(prev_masks=prev_masks, gt3D=gt3D, click_idx=click_idx, click_method=click_method, device=device)
            click_points.append(points_input)
            click_labels.append(labels_input)

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
    device = args.device
    print("device:", device)

    all_iou_list = []
    all_dice_list = []

    out_dice = dict()
    out_dice_all = OrderedDict()

    vis_root = os.path.join(os.path.dirname(__file__), args.vis_path)

    test_dataloader = get_test_dataloader(args)
    sam_model_tune = build_model(args)

    for batch_data in tqdm(test_dataloader, leave=False, colour='green'):
        image3D, gt3D, img_name = batch_data
        dataset_name = img_name[0].split('/')[-3]
        print(30*"-")
        print(f"Processing {img_name}")
        dataset = f"{dataset_name}_test" if args.data_type == "Ts" else f"{dataset_name}_val"
        save_root = os.path.join(vis_root, dataset)
        sz = image3D.size()
        if sz[2] < args.crop_size or sz[3] < args.crop_size or sz[4] < args.crop_size:
            print("[ERROR] wrong size", sz, "for", img_name)

        pred_path = os.path.join(
            save_root,
            os.path.basename(img_name[0]).replace(
                ".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"
            ),
        )

        if os.path.exists(pred_path):
            iou_list, dice_list = [], []
            for iter in range(args.num_clicks):
                curr_pred_path = os.path.join(
                    save_root,
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

            os.makedirs(save_root, exist_ok=True)
            points = [point_.cpu().numpy() for point_ in points]
            labels = [label_.cpu().numpy() for label_ in labels]
            pt_info = dict(points=points, labels=labels)
            pt_path = os.path.join(
                save_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pt.pkl")
            )
            pickle.dump(pt_info, open(pt_path, "wb"))
            for idx, pred3D in enumerate(seg_mask_list):
                out = sitk.GetImageFromArray(pred3D)
                sitk.WriteImage(
                    out,
                    os.path.join(
                        save_root,
                        os.path.basename(img_name[0]).replace(
                            ".nii.gz", f"_pred{idx}.nii.gz"
                        ),
                    ),
                )

        per_iou = max(iou_list)
        all_iou_list.append(per_iou)
        all_dice_list.append(max(dice_list))
        print(f"Dice Score after each of the {args.num_clicks} clicks: {dice_list}")
        out_dice[img_name] = max(dice_list)
        cur_dice_dict = OrderedDict()
        for i, dice in enumerate(dice_list):
            cur_dice_dict[f"{i}"] = dice
        out_dice_all[img_name[0]] = cur_dice_dict

    print("Mean IoU: ", sum(all_iou_list) / len(all_iou_list))
    print("Mean Dice: ", sum(all_dice_list) / len(all_dice_list))

    print(f"Preds saved to: {save_root}")

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        dataset = k.split("/")[-3]
        final_dice_dict[dataset] = OrderedDict()
    for k, v in out_dice_all.items():
        dataset = k.split("/")[-3]
        final_dice_dict[dataset][k] = v

    save_name = f"{vis_root}/out_dice.py"
    if args.split_num > 1:
        save_name = save_name.replace(".py", f"_s{args.split_num}i{args.split_idx}.py")

    print(f"Dice Scores saved to: {save_name}")
    with open(save_name, "w") as f:
        f.writelines(f"# mean dice: \t{np.mean(all_dice_list)}\n")
        f.writelines("dice_Ts = {")
        for k, v in out_dice.items():
            f.writelines(f"'{str(k[0])}': {v},\n")
        f.writelines("}")

    with open(save_name.replace(".py", ".json"), "w") as f:
        json.dump(final_dice_dict, f, indent=4)

    print("Done")

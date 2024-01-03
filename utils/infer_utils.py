import numpy as np
import torch
import torch.nn.functional as F

# def postprocess_masks(low_res_masks, image_size, original_size):
#     ori_h, ori_w = original_size
#     masks = F.interpolate(
#         low_res_masks,
#         (image_size, image_size),
#         mode="bilinear",
#         align_corners=False,
#     )
#     if args.ft2d and ori_h < image_size and ori_w < image_size:
#         top = (image_size - ori_h) // 2
#         left = (image_size - ori_w) // 2
#         masks = masks[..., top : ori_h + top, left : ori_w + left]
#         pad = (top, left)
#     else:
#         masks = F.interpolate(
#             masks, original_size, mode="bilinear", align_corners=False
#         )
#         pad = None
#     return masks, pad


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


import numpy as np

def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum
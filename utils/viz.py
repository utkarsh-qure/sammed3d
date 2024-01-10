import os
import json
import matplotlib.pyplot as plt
import numpy as np


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_dice_scores_json(json_file_path:str):
    with open(json_file_path, "r") as f:
        data = json.load(f)
    click_scores = {}
    for dataset_name in data.keys():
        click_scores[dataset_name] = {
            os.path.basename(scan): {click_idx: [] for click_idx in range(10)}
            for scan in data[dataset_name]
        }
        for scan, scores in data[dataset_name].items():
            for click_idx, score in scores.items():
                click_scores[dataset_name][os.path.basename(scan)][int(click_idx)] = score
    
    return click_scores


def plot_boxplots(click_scores, save_path):
    plt.figure(figsize=(10, 6))
    data = [
        [
            scan_scores[click_idx]
            for scan_scores in click_scores.values()
        ]
        for click_idx in range(10)
    ]
    plt.boxplot(
        data,
        labels=[f"click {click_idx+1}" for click_idx in range(10)],
        showfliers=False,
    )

    # plt.xlabel('click index')
    plt.ylabel("dice score")
    plt.title("Boxplot of Dice Scores by click index")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


def plot_dice_score_subplots(click_scores, save_path):
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle("Dice Score distribution for each click", fontsize=16)

    for click_idx in range(10):
        row = click_idx // 2
        col = click_idx % 2

        scores = [
            scan_scores[click_idx]
            for scan_scores in click_scores.values()
        ]

        axs[row, col].hist(scores, bins=20, alpha=0.5)
        axs[row, col].set_title(f"click {click_idx+1}")
        axs[row, col].set_xlabel("dice score")
        axs[row, col].set_ylabel("frequency")

        axs[row, col].axvline(
            np.mean(scores), color="red", linestyle="dashed", linewidth=1
        )
        axs[row, col].text(
            0.6, 0.9, f"mean: {np.mean(scores):.3f}", transform=axs[row, col].transAxes
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


root = "/home/users/utkarsh.singh/qct/medsam/SAM-Med3D/results"
task_name = "dsb_union_2reader_ft_turbo"

json_file_path = f"{root}/{task_name}/out_dice.json"
click_scores = parse_dice_scores_json(json_file_path)

dataset_name = "dsb"
plots_directory = f"{root}/{task_name}/plots/{dataset_name}"
create_directory_if_not_exists(plots_directory)

plot_boxplots(click_scores=click_scores[dataset_name], save_path=f"{plots_directory}/dice_score_boxplots.png")
plot_dice_score_subplots(click_scores=click_scores[dataset_name], save_path=f"{plots_directory}/dice_score_distribution_with_clicks.png")

import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from monai.losses import DiceCELoss

from utils.data_paths import img_datas
from build import get_dataloaders, build_model
from utils.click_method import get_next_click3D_torch_2

logger = logging.getLogger(__name__)

click_methods = {
    "random": get_next_click3D_torch_2,
}


class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        init_ckpt_path = ""
        if not args.train_from_scratch:
            init_ckpt_path = os.path.join(
                self.args.work_dir, self.args.task_name, "sam_model_latest.pth"
            )
            init_ckpt_path = (
                init_ckpt_path
                if os.path.exists(init_ckpt_path)
                else "./ckpt/sam_med3d_turbo.pth"
            )  # original: "./ckpt/sam_med3d.pth"
        self.init_checkpoint(ckp_path=init_ckpt_path)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    def set_optimizer(self):
        sam_model = self.model

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": sam_model.image_encoder.parameters()
                },  # , 'lr': self.args.lr * 0.1},
                {
                    "params": sam_model.prompt_encoder.parameters(),
                    "lr": self.args.lr * 0.1,
                },
                {
                    "params": sam_model.mask_decoder.parameters(),
                    "lr": self.args.lr * 0.1,
                },
            ],
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
        )

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.args.step_size, self.args.gamma
            )
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.args.step_size[0], self.args.gamma
            )
        elif self.args.lr_scheduler == "coswarm":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        if last_ckpt:
            self.model.load_state_dict(last_ckpt["model_state_dict"])
            print(f"SAM-Med3D size: {sum(p.numel() for p in self.model.parameters())}")
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt["epoch"]
                self.optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
                self.lr_scheduler.load_state_dict(last_ckpt["lr_scheduler_state_dict"])
                self.losses = last_ckpt["losses"]
                self.dices = last_ckpt["dices"]
                self.best_loss = last_ckpt["best_loss"]
                self.best_dice = last_ckpt["best_dice"]
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "used_datas": img_datas,
            },
            os.path.join(self.args.model_save_path, f"sam_model_{describe}.pth"),
        )

    def batch_forward(
        self, sam_model, image_embedding, gt3D, low_res_masks, points=None
    ):
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(self.args.device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(
            low_res_masks, size=gt3D.shape[-3:], mode="trilinear", align_corners=False
        )
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](
            prev_masks, gt3D
        )

        points_co = torch.cat(batch_points, dim=0).to(self.args.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(self.args.device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(self.args.device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        # initialise mask before first click
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)

        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(
                self.args.img_size // 4,
                self.args.img_size // 4,
                self.args.img_size // 4,
            ),
        )

        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            ## uncomment for (32, 128, 128) image
            # low_res_masks = F.interpolate(low_res_masks, size=(low_res_masks.shape[2]//4, low_res_masks.shape[3], low_res_masks.shape[4])) # low_res_mask along z dim should be 1/4th of other 2 dims, based on img size
            # breakpoint()

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks, points=None
                )
            else:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D,
                    low_res_masks,
                    points=[points_input, labels_input],
                )
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss

    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = mask_pred > mask_threshold
            mask_gt = mask_gt > 0

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = prev_masks > 0.5
        true_masks = gt3D > 0
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch_simple(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0
        self.model.train()
        sam_model = self.model

        self.optimizer.zero_grad()
        step_loss = 0

        tbar = tqdm(self.dataloaders)
        for step, (image3D, gt3D) in enumerate(tbar):
            image3D = image3D.to(self.args.device)
            gt3D = gt3D.to(self.args.device).type(torch.long)

            image_embedding = sam_model.image_encoder(image3D)

            self.click_points = []
            self.click_labels = []

            prev_masks, loss = self.interaction(
                sam_model, image_embedding, gt3D, num_clicks=num_clicks
            )

            epoch_loss += loss.item()
            cur_loss = loss.item()

            loss /= self.args.accumulation_steps
            loss.backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
                epoch_dice += print_dice  # Accumulate dice scores for the entire epoch

                logger.info(
                    f"Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}"
                )
                if print_dice > self.step_best_dice:
                    self.step_best_dice = print_dice
                    if print_dice > 0.6:
                        self.save_checkpoint(
                            epoch,
                            sam_model.state_dict(),
                            describe=f"{epoch}_step_dice:{print_dice}_best",
                        )
                if print_loss < self.step_best_loss:
                    self.step_best_loss = print_loss
            else:
                step_loss += cur_loss

        epoch_loss /= step
        epoch_dice /= step // self.args.accumulation_steps

        return epoch_loss, epoch_iou, epoch_dice

    def eval_epoch(self, epoch, num_clicks):
        return 0

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel("Epoch")
        plt.ylabel(f"{save_name}")
        plt.savefig(os.path.join(self.args.model_save_path, f"{save_name}.png"))
        plt.close()

    def train(self):
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f"Epoch: {epoch}/{self.args.num_epochs - 1}")

            # num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice = self.train_epoch_simple(
                epoch, num_clicks=11
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.losses.append(epoch_loss)
            self.dices.append(epoch_dice)
            print(f"EPOCH: {epoch}, Loss: {epoch_loss}")
            print(f"EPOCH: {epoch}, Dice: {epoch_dice}")
            logger.info(f"Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}")

            state_dict = self.model.state_dict()

            # save latest checkpoint
            self.save_checkpoint(epoch, state_dict, describe="latest")

            # save train loss best checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(epoch, state_dict, describe="loss_best")

            # save train dice best checkpoint
            if epoch_dice > self.best_dice:
                self.best_dice = epoch_dice
                self.save_checkpoint(epoch, state_dict, describe="dice_best")

            self.plot_result(self.losses, "Dice + Cross Entropy Loss", "Loss")
            self.plot_result(self.dices, "Dice", "Dice")

        logger.info(
            "====================================================================="
        )
        logger.info(f"Best loss: {self.best_loss}")
        logger.info(f"Best dice: {self.best_dice}")
        logger.info(f"Total loss: {self.losses}")
        logger.info(f"Total dice: {self.dices}")
        logger.info(
            "====================================================================="
        )
        logger.info(f"args : {self.args}")
        logger.info(f"Used datasets : {img_datas}")
        logger.info(
            "====================================================================="
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", type=str, default="dsb_union_2reader_ft_turbo"
    )  # last: dsb_union_2reader_ft_turbo
    parser.add_argument("--click_type", type=str, default="random")
    parser.add_argument("--multi_click", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default="vit_b_ori")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/cache/fast_data_nas8/utkarsh/training/sammed3d_workdir", # qure servers
        # default="/fast_data_2d_1/utkarsh/training/sammed3d_workdir", # e2e
    )

    # train
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true", default=False)

    # lr_scheduler
    parser.add_argument("--lr_scheduler", type=str, default="multisteplr")
    parser.add_argument("--step_size", type=list, default=[120, 180])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    args = parser.parse_args()

    args.device = "cuda:0"
    print(f"on device: {args.device}")

    args.model_save_path = os.path.join(args.work_dir, args.task_name)
    os.makedirs(args.model_save_path, exist_ok=True)

    args.train_from_scratch = (
        False  # train_from_scratch is True for any img not of the shape (128, 128, 128)
    )

    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    dataloaders = get_dataloaders(args)
    # dataloaders = get_dataloaders_32(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()


if __name__ == "__main__":
    main()

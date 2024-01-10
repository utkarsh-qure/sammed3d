import os
from glob import glob
import torch
from torch.utils.data import DataLoader

import torchio as tio

from segment_anything.build_sam3D import sam_model_registry3D

from utils.data_paths import img_datas
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
    if args.checkpoint_path is not None:
        model_dict = torch.load(args.checkpoint_path, map_location=args.device)
        state_dict = model_dict["model_state_dict"]
        sam_model.load_state_dict(state_dict)
    return sam_model


def get_dataloaders(args):
    train_val_transforms = tio.Compose(
        [
            tio.ToCanonical(),
            # tio.Resample((1, 1, 1)),
            tio.CropOrPad(
                mask_name="label",
                target_shape=(args.img_size, args.img_size, args.img_size),
            ),  # crop only object region
            tio.RandomFlip(axes=(0, 1, 2)),
        ]
    )
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        mode="train",
        data_type="Tr",
        transform=train_val_transforms,
        threshold=100,
    )

    val_dataset = Dataset_Union_ALL(
        paths=img_datas,
        mode="test",
        data_type="Val",
        transform=train_val_transforms,
        threshold=100,
    )

    # normal pytorch DataLoader with prefetch_generator:BackgroundGenerator
    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=None,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = Union_Dataloader(
        dataset=val_dataset,
        sampler=None,
        batch_size=args.batch_size//2,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


def get_dataloaders_32(args):
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose(
            [
                tio.ToCanonical(),
                # tio.Resample((1, 1, 1)),
                tio.CropOrPad(
                    mask_name="label",
                    target_shape=(args.img_size // 4, args.img_size, args.img_size),
                ),  # crop only object region
                tio.RandomFlip(axes=(0, 1, 2)),
            ]
        ),
        threshold=100,
    )

    # normal pytorch DataLoader with prefetch_generator:BackgroundGenerator
    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=None,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader


def get_test_dataloader(args):
    # all_dataset_paths = glob(args.test_data_path)
    all_dataset_paths = glob(os.path.join(args.test_data_path, "*"))
    # all_dataset_paths = glob(os.path.join(args.test_data_path, "*", "*"))

    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.Resample((1, 1, 1)),
        tio.CropOrPad(
            mask_name="label",
            target_shape=(args.crop_size, args.crop_size, args.crop_size),
        ),
    ]

    test_dataset = Dataset_Union_ALL(
        paths=all_dataset_paths,
        mode="test",
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

    return test_dataloader
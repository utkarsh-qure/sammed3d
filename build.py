import torchio as tio

from segment_anything.build_sam3D import sam_model_registry3D

from utils.data_paths import img_datas
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose(
            [
                tio.ToCanonical(),
                # tio.Resample((1, 1, 1)),
                tio.CropOrPad(
                    mask_name="label",
                    target_shape=(args.img_size, args.img_size, args.img_size),
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

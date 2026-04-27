from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .constants import IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_imagefolder_loaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    val_size: float = 0.2,
):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if train_dir.exists() and val_dir.exists():
        datasets_by_split = {
            "train": datasets.ImageFolder(train_dir, transform=build_train_transform(image_size)),
            "val": datasets.ImageFolder(val_dir, transform=build_eval_transform(image_size)),
        }
    else:
        all_data_dir = data_dir / "all_data"
        if not all_data_dir.exists():
            raise FileNotFoundError(
                f"Expected either {train_dir} and {val_dir}, or {all_data_dir} for train/val splitting."
            )
        train_val_dataset = datasets.ImageFolder(all_data_dir, transform=build_train_transform(image_size))
        train_indices, val_indices = train_test_split(
            list(range(len(train_val_dataset))),
            test_size=val_size,
            random_state=seed,
            stratify=train_val_dataset.targets,
        )
        val_dataset = datasets.ImageFolder(all_data_dir, transform=build_eval_transform(image_size))
        datasets_by_split = {
            "train": Subset(train_val_dataset, train_indices),
            "val": Subset(val_dataset, val_indices),
        }

    if test_dir.exists():
        datasets_by_split["test"] = datasets.ImageFolder(test_dir, transform=build_eval_transform(image_size))
    else:
        datasets_by_split["test"] = datasets_by_split["val"]

    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return datasets_by_split, loaders

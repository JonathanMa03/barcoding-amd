from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_oct_c8_transforms():
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def get_oct_c8_dataloaders(
    data_dir,
    batch_size=32,
    num_workers=0
):
    data_dir = Path(data_dir)

    train_transform, eval_transform = get_oct_c8_transforms()

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        data_dir / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.classes

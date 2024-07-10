import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import logging


def data_loader() -> tuple[DataLoader, DataLoader]:

    logging.basicConfig(level=logging.INFO)
    # Define the data directory
    DATA_DIR = "./data"

    # Define the transformations for the training and test sets
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Download and prepare the Food-101 dataset
    train_dataset = datasets.Food101(
        root=DATA_DIR, split="train", download=True, transform=transform_train
    )
    test_dataset = datasets.Food101(
        root=DATA_DIR, split="test", download=True, transform=transform_test
    )

    # Define the DataLoader for training and test sets
    bs = 1500
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        pin_memory=True,
        num_workers=15,
        prefetch_factor=2,
        persistent_workers=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=15)

    return train_loader, test_loader


if __name__ == "__main__":
    data_loader()

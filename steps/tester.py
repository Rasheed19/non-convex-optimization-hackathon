import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from PIL import Image

from utils.test_helper import score, load_checkpoint, reset, count_parameters

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def test(model, sample):
    model.eval()

    with torch.no_grad():

        img = sample["img"].to(DEVICE)
        label = sample["id"].to(DEVICE)
        pred = model(img)
        num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)

    return num_correct.item()


class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data["path"])

    def __getitem__(self, idx):

        sample = dict()
        idx = idx + 11362
        img_path = self.data["path"][idx]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        sample["img"] = image
        sample["id"] = self.data["id"][idx]
        sample["label"] = self.data["label"][idx]

        return sample


def model_tester(saved_model_name: str) -> float:

    test_data_path = "./test/test.pkl"

    with open(test_data_path, "rb") as file:
        test_data = pickle.load(file)

    batch_size = 256

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_dataset = TestDataset(test_data, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    reset(0)

    model = load_checkpoint(f"./models/{saved_model_name}", DEVICE)

    num_params = count_parameters(model)
    if num_params > 5000000:  # The given EXAMPLE model has 20 million parameter
        raise ValueError("Cannot have more than 5 million parameters!")

    avg_te_correct = 0
    for sample in test_loader:
        te_correct = test(model, sample)
        avg_te_correct += te_correct / len(test_dataset)

    return avg_te_correct * 100

import json
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import os
from torchvision.io import read_image


class ImageDataset(Dataset):
    """
    A Dataset class to create Pytorch Dataset instance which can be passed on to DataLoader
    This method can perform transformations when `image_transform` and `target_transform` are not `None`
    """

    def __init__(
        self,
        img_dir,
        target_dir,
        image_transform=None,
        target_transform=None,
        mode="train",
    ):
        self.img_labels_path_list = [
            json.load(open(os.path.join(target_dir, path)))
            for path in os.listdir(target_dir)
        ]
        self.img_path_list = [
            os.path.join(img_dir, path) for path in os.listdir(img_dir)
        ]
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self):
        return len(self.img_labels_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_path_list[idx])
        label = self.img_labels_path_list[idx]
        if self.image_transform:
            image = self.image_transform(image, self.mode)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

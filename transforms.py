import torch
from torchvision.transforms import transforms

INPUT_SIZE = (500, 250)

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


def img_transform(img, mode):
    if mode == "train":
        return data_transforms["train"](img)
    else:
        return data_transforms["val"](img)


def target_transform(val):
    absolute_reading = val["absolute_reading"]
    relative_reading = val["relative_reading"]
    max_val, min_val = val["range"][1], val["range"][0]
    return torch.Tensor(([100 * (absolute_reading - min_val) / (max_val - min_val)]))

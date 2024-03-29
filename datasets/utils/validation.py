# Get Train Validation
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.transforms as transforms

from utils import create_if_not_exists


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, targets: np.ndarray,
                 transform: transforms = None, target_transform: transforms = None, image_size: int = 225) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, np.ndarray):
            if np.max(img) < 2:
                img = Image.fromarray(np.uint8(img * self.image_size))
            else:
                img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_train_val(train, test_transform, dataset, image_size=255, val_perc=0.1):
    dataset_length = train.data.shape[0]
    directory = f'datasets/val_permutations/{image_size}/'
    create_if_not_exists(directory)
    file_name = dataset + '.pt'
    if os.path.exists(directory + file_name):
        perm = torch.load(directory + file_name)
    else:
        perm = torch.randperm(dataset_length)
        torch.save(perm, directory + file_name)
    train.data = train.data[perm]
    train.targets = np.array(train.targets)[perm]
    test_dataset = ValidationDataset(train.data[:int(
        val_perc * dataset_length)], train.targets[:int(val_perc * dataset_length)], transform=test_transform, image_size=image_size)
    train.data = train.data[int(val_perc * dataset_length):]
    train.targets = train.targets[int(val_perc * dataset_length):]

    return train, test_dataset

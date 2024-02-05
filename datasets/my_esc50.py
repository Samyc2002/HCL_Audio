# My ESC50 dataset
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from .utils.continual_dataset import ContinualDataset, get_previous_train_loader, store_masked_loaders
from .utils.validation import get_train_val
from augmentations import get_aug
import librosa
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MYESC50():
    def __init__(self, data, targets, transform):
        self.data = np.stack([np.array(image) for image in data])
        self.targets = np.array(targets)
        self.transform = transform

    def __getitem__(self, index):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
#         print("Image Shape", np.shape(img), "Target", target)
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        img, img1, not_aug_img = self.transform(original_img)

        if hasattr(self, 'logits'):
            return (img, img1, not_aug_img), target, self.logits[index]

        return (img, img1, not_aug_img),  target

    def __len__(self):
        return len(self.targets)


def my_esc50(train=False, transform=None):
    """
    Preprocesses the ESC50 dataset.
    """
    file_path = "E:\\Projects\\ContinuousLearning\\HCL_Audio\\assets\esc50\\"

    files = os.listdir(file_path)
    data, targets = [], []

    for filename in files:
        y, sr = librosa.load(file_path + filename, sr=None)

        S_dB = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S_dB)
        # librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

        # Convert to PIL image
        S_dB -= S_dB.min()
        S_dB *= (255.0 / S_dB.max())
        S_img = Image.fromarray(S_dB.astype(np.uint8)).convert(
            'RGB').resize((255, 255))

        data.append(S_img)
        targets.append(int(Path(filename).stem.split("-")[-1]))

    return MYESC50(data, targets, transform)


class ESC50(ContinualDataset):
    """
    Returns a structured ESC50 class
    """

    NAME = "esc50"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 5
    N_TASKS = 10

    def __init__(self):
        super(ESC50, self).__init__()

    def get_data_loaders(self):
        esc_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        transform = get_aug(train=True, mean_std=esc_norm,
                            name="simsiam", image_size=32, cl_default=True)
        test_transform = get_aug(train=False, train_classifier=False,
                                 mean_std=esc_norm, name="simsiam", image_size=32, cl_default=True)

        train_dataset = my_esc50(train=True, transform=transform)
        memory_dataset = my_esc50(train=True, transform=test_transform)

        train_dataset, test_dataset = get_train_val(
            train_dataset, test_transform, self.NAME)
        memory_dataset, _ = get_train_val(
            memory_dataset, test_transform, self.NAME)

        train, memory, test = store_masked_loaders(
            train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test

    def get_transform(self):
        esc_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(*esc_norm)
             ]
        )

        return transform

    def not_aug_dataloader(self, batch_size):
        esc_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*esc_norm)])

        train_dataset = my_esc50(train=True, transform=transform)
        train_loader = get_previous_train_loader(
            train_dataset, batch_size, self)

        return train_loader

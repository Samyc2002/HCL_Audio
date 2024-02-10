# My FSC dataset
from utils import create_if_not_exists
import numpy as np
import h5py
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from .utils.continual_dataset import ContinualDataset, get_previous_train_loader, store_masked_loaders
from .utils.validation import get_train_val
from augmentations import get_aug
import librosa
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MYFSC():
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
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        img, img1, not_aug_img = self.transform(original_img)

        if hasattr(self, 'logits'):
            return (img, img1, not_aug_img), target, self.logits[index]

        return (img, img1, not_aug_img),  target

    def __len__(self):
        return len(self.targets)


def my_fsc(name, transform=None):
    """
    Preprocesses the FSC dataset.
    """
    directory = f'data/{name}/'
    create_if_not_exists(directory)

    if os.path.exists(directory + "image_data.h5") and os.path.exists(directory + "target_data.h5"):
        # Return from file itself
        h5image = h5py.File(directory + "image_data.h5", "r")
        image_data = h5image[name][:]
        h5image.close()
        h5target = h5py.File(directory + "target_data.h5", "r")
        target_data = h5target[name][:]
        h5target.close()
        # print(image_data.shape)

        return MYFSC(image_data, target_data, transform)
    else:
        file_path = f"E:\\Projects\\ContinuousLearning\\HCL_Audio\\assets\{name}\\"

        files = os.listdir(file_path)
        data, targets = [], []

        for filename in files:
            y, sr = librosa.load(file_path + filename, sr=44100)

            S_dB = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=32, fmax=12000, hop_length=512)
            S_dB = librosa.power_to_db(S_dB)
            # librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

            # Ensure the shape is [n_mels, time_frames] without singleton dimensions
            S_dB = np.squeeze(S_dB)

            # Manually reshape the array to have more than one value per channel
            # S_dB = np.repeat(S_dB[:, np.newaxis], 3, axis=-1)

            # Convert to PIL image
            S_dB -= S_dB.min()
            S_dB *= (255.0 / S_dB.max())
            S_img = Image.fromarray(S_dB.astype(np.uint8)).convert(
                'RGB').resize((255, 255))

            data.append(S_img)
            targets.append(int(Path(filename).stem.split("_")[0]) - 1)

        # Cache to file
        h5image = h5py.File(directory + "image_data.h5", "w")
        image_data = h5image.create_dataset(name, data=data)
        h5image.close()
        h5target = h5py.File(directory + "target_data.h5", "w")
        target_data = h5target.create_dataset(name, data=targets)
        h5target.close()

        return MYFSC(data, targets, transform)


class FSC(ContinualDataset):
    """
    Returns a structured ESC50 class
    """

    NAME = "fsc"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 3
    N_TASKS = 9

    def __init__(self):
        super(FSC, self).__init__()

    def get_data_loaders(self):
        esc_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        transform = get_aug(train=True, mean_std=esc_norm,
                            name="simsiam", image_size=32, cl_default=True)
        test_transform = get_aug(train=False, train_classifier=False,
                                 mean_std=esc_norm, name="simsiam", image_size=32, cl_default=True)

        train_dataset = my_fsc(self.NAME, transform=transform)
        memory_dataset = my_fsc(self.NAME, transform=test_transform)

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

        train_dataset = my_fsc(self.NAME, train=True, transform=transform)
        train_loader = get_previous_train_loader(
            train_dataset, batch_size, self)

        return train_loader

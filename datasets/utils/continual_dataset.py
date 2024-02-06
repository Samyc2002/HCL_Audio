# Continual Dataset
from abc import abstractmethod
from torch.utils.data import DataLoader
from torch import nn as nn
import numpy as np

BATCH_SIZE = 2


class ContinualDataset:
    """
    Continual Learning Settings
    """

    NAME = None
    SETTING = None
    # Number of classes
    N_CLASSES_PER_TASK = None
    # Number of tasks from a class
    N_CLASSES = None
    TRANSFORM = None

    def __init__(self):
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.memory_loaders = []
        self.train_loaders = []
        self.i = 0

    @abstractmethod
    def get_data_loaders(self):
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size):
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone():
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform():
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss():
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform():
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform():
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders(train_dataset, test_dataset, memory_dataset, setting):
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i, np.array(
        train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i, np.array(
        test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    memory_dataset.data = memory_dataset.data[train_mask]
    memory_dataset.targets = np.array(memory_dataset.targets)[train_mask]

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)
    memory_loader = DataLoader(
        memory_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    setting.test_loaders.append(test_loader)
    setting.train_loaders.append(train_loader)
    setting.memory_loaders.append(memory_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, memory_loader, test_loader


def store_masked_label_loaders(train_dataset, test_dataset, memory_dataset, setting):
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i, np.array(
        train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i, np.array(
        test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    memory_dataset.data = memory_dataset.data[train_mask]
    memory_dataset.targets = np.array(memory_dataset.targets)[train_mask]

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)
    memory_loader = DataLoader(
        memory_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    setting.test_loaders.append(test_loader)
    setting.train_loaders.append(train_loader)
    setting.memory_loaders.append(memory_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, memory_loader, test_loader


def store_domain_loaders(train_dataset, test_dataset, memory_dataset, setting):
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    memory_loader = DataLoader(memory_dataset,
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    setting.test_loaders.append(test_loader)
    setting.train_loaders.append(train_loader)
    setting.memory_loaders.append(memory_loader)
    setting.train_loader = train_loader

    # setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, memory_loader, test_loader


def get_previous_train_loader(train_dataset, batch_size, setting):
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i -
                                setting.N_CLASSES_PER_TASK, np.array(
                                    train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

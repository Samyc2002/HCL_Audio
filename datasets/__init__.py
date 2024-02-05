# Dataset utils
from .my_esc50 import ESC50

NAMES = {
    ESC50.NAME: ESC50
}
N_CLASSES = {'esc50': 10}
BACKBONES = {
    'esc50': ["resnet18", "resnet18", "resnet18", "resnet18", "resnet18", "resnet18", "resnet18", "resnet18", "resnet18", "resnet18"]
}


def get_dataset():
    # Generate the Dataset
    return NAMES["esc50"]()

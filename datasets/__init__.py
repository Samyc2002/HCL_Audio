# Dataset utils
from .my_esc50 import ESC50

NAMES = {
    ESC50.NAME: ESC50
}
N_CLASSES = {'esc50': 10}
BACKBONES = {
    # + ["resnet34"] * 2 + ["vgg16.tv_in1k"] * 2 + ["mixer_b16_224.miil_in21k_ft_in1k"] * 2
    'esc50': ["resnet18"] * 10
}


def get_dataset():
    # Generate the Dataset
    return NAMES["esc50"]()

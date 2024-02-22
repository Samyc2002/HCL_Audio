# Dataset utils
from .my_esc10 import ESC10
from .my_esc50 import ESC50
from .my_dcase19 import DCASE19
from .my_fsc import FSC

NAMES = {
    ESC10.NAME: ESC10,
    ESC50.NAME: ESC50,
    DCASE19.NAME: DCASE19,
    FSC.NAME: FSC
}


N_CLASSES = {
    'esc10': 5,
    'esc50': 10,
    "dcase19": 3,
    "fsc": 9
}

BACKBONES = {
    'esc10': [
        "resnet18",
        "resnet34",
        "vgg16.tv_in1k",
        "mixer_b16_224.miil_in21k_ft_in1k",
        "vit_base_patch16_224"
    ],
    'esc50': [
        "resnet18",
        "resnet18",
        "resnet34",
        "resnet34",
        "vgg16.tv_in1k",
        "vgg16.tv_in1k",
        "mixer_b16_224.miil_in21k_ft_in1k",
        "mixer_b16_224.miil_in21k_ft_in1k",
        "vit_base_patch16_224",
        "vit_base_patch16_224"
    ],
    "dcase19": [
        "resnet18",
        "vgg16.tv_in1k",
        "mixer_b16_224.miil_in21k_ft_in1k"
    ],
    "fsc": [
        "resnet18",
        "resnet18",
        "resnet34",
        "resnet34",
        "vgg16.tv_in1k",
        "vgg16.tv_in1k",
        "mixer_b16_224.miil_in21k_ft_in1k",
        "mixer_b16_224.miil_in21k_ft_in1k",
        "vit_base_patch16_224"
    ],
}


def get_dataset(args):
    # Generate the Dataset
    return NAMES[args.dataset]()

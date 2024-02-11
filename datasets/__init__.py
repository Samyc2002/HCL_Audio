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
N_CLASSES = {'esc10': 5, 'esc50': 10, "dcase19": 3, "fsc": 9}
BACKBONES = {
    'esc10': ["resnet18"] * 2 + ["resnet34"] * 1 + ["vgg16.tv_in1k"] * 1 + ["mixer_b16_224.miil_in21k_ft_in1k"] * 1,
    'esc50': ["resnet18"] * 4 + ["resnet34"] * 2 + ["vgg16.tv_in1k"] * 2 + ["mixer_b16_224.miil_in21k_ft_in1k"] * 2,
    "dcase19": ["resnet18", "vgg16.tv_in1k", "mixer_b16_224.miil_in21k_ft_in1k"],
    "fsc": ["resnet18"] * 3 + ["resnet34"] * 2 + ["vgg16.tv_in1k"] * 2 + ["mixer_b16_224.miil_in21k_ft_in1k"] * 2,
}


def get_dataset(args):
    # Generate the Dataset
    return NAMES[args.dataset]()

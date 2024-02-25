import timm

from datasets import BACKBONES, N_CLASSES
from .simsiam import SimSiam
from .distil import Distil
from utils.losses import LabelSmoothing


def get_backbone(args, task_id=0):
    # Get the model backbone
    backbone = BACKBONES[args.dataset][task_id]

    NUM_CLASSES = {
        "esc10": 10,
        "esc50": 50,
        "fsc10": 27,
        "dcase19": 9
    }

    net = eval(
        f"timm.create_model('{backbone}', pretrained=True, num_classes={NUM_CLASSES[args.dataset]})")
    print("Backbone changed to ", backbone)

    net.n_classes = N_CLASSES[args.dataset]
    if backbone == "resnet18":
        net.output_dim = net.fc.in_features
    elif backbone == "resnet34":
        net.output_dim = net.fc.in_features
    elif backbone == "vgg16.tv_in1k":
        net.output_dim = net.head.fc.in_features
    elif backbone == "mixer_b16_224.miil_in21k_ft_in1k":
        net.output_dim = net.head.in_features
    elif backbone == "vit_base_patch16_224":
        net.output_dim = net.head.in_features

    return net


def get_model(device, dataset, transform, args, global_model=None, task_id=0):
    # Get the model
    loss = LabelSmoothing(smoothing=0.1)

    backbone = SimSiam(get_backbone(args, task_id=task_id)).to(device)
    backbone.projector.set_layers(2)

    return Distil(backbone, loss, dataset, transform, global_model, args)

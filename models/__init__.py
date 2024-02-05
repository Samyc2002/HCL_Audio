import os
import importlib

from datasets import BACKBONES, N_CLASSES
from .simsiam import SimSiam
from .distil import Distil
from utils.losses import LabelSmoothing
from torchvision.models import resnet18


def get_backbone(task_id=0):
    # Get the model backbone
    backbone = BACKBONES["esc50"][task_id]

    net = eval(f"{backbone}(num_classes=N_CLASSES['esc50'])")
    print("Backbone changed to ", backbone)

    net.n_classes = N_CLASSES["esc50"]
    net.output_dim = net.fc.in_features

    return net


def get_model(dataset, transform, global_model=None, task_id=0):
    # Get the model
    loss = LabelSmoothing(smoothing=0.1)

    backbone = SimSiam(get_backbone(task_id=task_id)).to("cuda:0")
    backbone.projector.set_layers(2)

    return Distil(backbone, loss, dataset, transform, global_model)

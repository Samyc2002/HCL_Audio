# Continual Model
import torch.nn as nn
import torch

from models.optimizers import get_optimizer


class ContinualModel(nn.Module):
    """
    Continual Learning Model
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone, loss, dataset, transform):
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.net = nn.DataParallel(self.net)
        self.loss = loss
        self.transform = transform
        self.dataset = dataset

        self.opt = get_optimizer(
            "sgd", self.net, lr=0.03, momentum=0.9, weight_decay=0.0005, cl_default=True)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net.module.backbone.forward(x)

    def observe(self, inputs, labels, not_aug_inputs):
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

# Distil Model
import torch

from utils.losses import KL_div_Loss
from utils.buffer import Buffer
from .utils.continual_model import ContinualModel


class Distil(ContinualModel):
    NAME = "distil"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, len_train_loader, transform, global_model):
        super(Distil, self).__init__(
            backbone, loss, len_train_loader, transform)
        self.global_model = global_model
        self.buffer = Buffer(200, self.device)
        self.global_model = global_model
        self.criterion_kl = KL_div_Loss(temperature=1.0).cuda()
        self.soft = torch.nn.Softmax(dim=1)

    def observe(self, inputs1, labels, inputs2, notaug_inputs, task_id):
        self.opt.zero_grad()
        inputs1, labels = inputs1.to(self.device), labels.to(self.device)
        inputs2 = inputs2.to(self.device)
        notaug_inputs = notaug_inputs.to(self.device)
        real_batch_size = inputs1.shape[0]

        if task_id:
            self.global_model.eval()
            outputs = self.net.module.backbone(inputs1)
            with torch.no_grad():
                outputs_teacher = self.global_model.net.module.backbone(
                    inputs1)

            penalty = 3.0 * self.criterion_kl(outputs, outputs_teacher)
            loss = self.loss(outputs, labels) + penalty
        else:
            outputs = self.net.module.backbone(inputs1)
            loss = self.loss(outputs, labels)

        if task_id:
            data_dict = {'loss': loss, 'penalty': penalty}
        else:
            data_dict = {'loss': loss, 'penalty': 0.}

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': 0.03})

        return data_dict

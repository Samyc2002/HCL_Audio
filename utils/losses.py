# Losses
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        target = target.to(torch.int64)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# Define KL divergence loss
class KL_div_Loss(nn.Module):
    """
    We use formulation of Hinton et. for KD loss.
    $T^2$ scaling is implemented to avoid gradient rescaling when using T!=1
    """

    def __init__(self, temperature):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(KL_div_Loss, self).__init__()
        self.temperature = temperature
        # print( "Setting temperature = {} for KD (Only Teacher)".format(self.temperature) )
        print("Setting temperature = {} for KD".format(self.temperature))

    def forward(self, y, teacher_scores):
        p = F.log_softmax(y / self.temperature, dim=1)  # Hinton formulation

        # p = F.log_softmax(y, dim=1) # Muller et. al used this.

        q = F.softmax(teacher_scores / self.temperature, dim=1)
        l_kl = F.kl_div(p, q, reduction='batchmean')
        return l_kl*(self.temperature**2)  # $T^2$ scaling is important

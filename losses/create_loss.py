import torch
import torch.nn as nn

from .convnext_small_perceptual import ConvnextSmallPerceptualLoss
from .fid_perceptualer import FIDPerceptual

class NonePerceptual(nn.Module):
    def forward(self, pred, target):
        return 0.0*torch.mean((pred - target)**2)

def init_perceptualer(config):
    if config["target"] == "convnext_small_perceptual":
        return ConvnextSmallPerceptualLoss(weight=config["weight"])
    elif config["target"] == "fid":
        return FIDPerceptual(weight=config["weight"])
    elif config["target"] == "none":
        return NonePerceptual()
    else:
        raise ValueError(f"Unknown target for perceptualer: {config['target']}")

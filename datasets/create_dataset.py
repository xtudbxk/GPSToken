import torch
import torch.nn as nn

from .adaptivegps import AdaptiveGPSDataset

def init_dataset(config):
    if config["target"] == "AdaptiveGPSDataset":
        return AdaptiveGPSDataset(**config["params"])
    else:
        raise ValueError(f"Unknown target {config['target']}")

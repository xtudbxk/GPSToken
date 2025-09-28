"""This file contains perceptual loss module using ConvNeXt-S.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import torch
import torch.nn.functional as F

from torchvision import models

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

 
class ConvnextSmallPerceptualLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

        self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()
        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Always in eval mode.
        self.eval()

        input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
        target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
        pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
        pred_target = self.convnext((target - self.imagenet_mean) / self.imagenet_std)
        loss = torch.nn.functional.mse_loss(
            pred_input,
            pred_target,
            reduction="mean")
    
        return self.weight*loss

import torch
import torch.nn as nn
import timm

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes
        )

        # Adapt for CIFAR-10 (32x32)
        self.model.patch_embed.img_size = (32, 32)

    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import ASPP


from .attention import ECA, CBAM


class DeepLabV3_Attention(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()

        # Backbone with dilation
        self.backbone = resnet50(
            weights=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True],
        )

        # Remove avgpool & fc
        self.backbone_layers = nn.Sequential(*list(self.backbone.children())[:-2])

        # Channel = 2048 at layer4 output
        self.eca = ECA(2048)

        # ASPP ⇒ output_channels = 256
        self.aspp = ASPP(2048, (12, 24, 36))

        # ASPP output channels = 256
        self.cbam = CBAM(256)

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[-2:]

        x = self.backbone_layers(x)
        x = self.eca(x)
        x = self.aspp(x)
        x = self.cbam(x)
        x = self.classifier(x)

        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return {"out": x}


def deeplabv3_resnet50_attention(
    *,
    num_classes: int = 1,
    pretrained_backbone: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs,
):
    return DeepLabV3_Attention(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone
    )

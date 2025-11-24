import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import ASPP

from .attention import CBAM, ECA


# --------------------------
# Simple Self-Attention (Non-local block)
# --------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        Q = self.query(x).view(B, -1, H * W)          # (B, C/8, N)
        K = self.key(x).view(B, -1, H * W)            # (B, C/8, N)
        V = self.value(x).view(B, -1, H * W)          # (B, C,   N)

        attention = torch.softmax(Q.permute(0, 2, 1) @ K, dim=-1)  # (B, N, N)
        out = V @ attention.permute(0, 2, 1)                        # (B, C, N)
        out = out.view(B, C, H, W)

        return self.gamma * out + x


# --------------------------
# Deeplabv3+ with Attention
# --------------------------
class DeepLabV3Plus_Attention(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=True):
        super().__init__()

        # Backbone (ResNet50)
        self.backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
            replace_stride_with_dilation=[False, True, True]
        )
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Low-level feature (layer2: 256 channels)
        self.low_level_conv = nn.Conv2d(256, 48, 1)

        # ASPP output: 256 channels
        self.aspp = ASPP(2048, [12, 24, 36])

        # ★ Self-Attention
        self.self_att = SelfAttention(256)

        # ★ CBAM (optional but recommended)
        self.cbam = CBAM(256)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # backbone forward
        low = self.backbone[:5](x)     # (B, 256, H/4, W/4)
        high = self.backbone[5:](low)  # (B, 2048, H/16, W/16)

        # ASPP
        x = self.aspp(high)

        # Apply Attention
        x = self.self_att(x)
        x = self.cbam(x)

        # Upsample
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        # Low-level projection
        low = self.low_level_conv(low)

        # Decoder concat
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)

        # Final upsample
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return {"out": x}
        
def deeplabv3_resnet50_attention(
    num_classes=1,
    pretrained_backbone=True,
    **kwargs,
):
    return DeepLabV3Plus_Attention(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone
    )

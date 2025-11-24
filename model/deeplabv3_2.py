import torch
import torch.nn as nn

# 내가 만든 attention 모델을 불러오기
from model.deeplabv3_attention import deeplabv3_resnet50_attention
from torchvision.models.resnet import ResNet50_Weights



def get_model(num_classes=1, pretrained=True):

    # pretrained 사용 여부
    backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None

    # Attention DeepLabV3 모델 생성
    model = deeplabv3_resnet50_attention(
        num_classes=num_classes,
        pretrained_backbone=backbone_weights,
    )

    return model

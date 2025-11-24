import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_model(num_classes=1, pretrained=True):
    """
    사전 학습된 DeepLabv3 (ResNet50 백본) 모델을 불러옴
    최종 출력 레이어를 주어진 num_classes에 맞게 수정
    """

    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
    else:
        weights = None
        
    model = deeplabv3_resnet50(weights=weights, aux_loss=True)
    
    model.classifier[4] = nn.Conv2d(
        256, # DeepLabv3 classifier의 마지막 in_channels
        num_classes, # (배경/건물 = 1)
        kernel_size=(1, 1),
        stride=(1, 1)
    )
    
    # 이것도 똑같이 1개 클래스로 변경
    model.aux_classifier[4] = nn.Conv2d(
        256, # DeepLabv3 aux_classifier의 마지막 in_channels
        num_classes, 
        kernel_size=(1, 1),
        stride=(1, 1)
    )
    
    return model
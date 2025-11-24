import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataset import SatelliteDataset
from model.u_net import UNet
from utils.utils import rle_encode

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    test_dataset = SatelliteDataset(csv_file='../data/test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # model 초기화 및 학습된 가중치 로드
    model = UNet().to(device)
    model.load_state_dict(torch.load('unet_model.pth')) # train.py 에서 저장한 모델 경로
    model.eval()

    result = []
    with torch.no_grad():
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            
            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    # Submission
    submit = pd.read_csv('../data/sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./submit.csv', index=False)
    print("Submission file created: submit.csv")

if __name__ == '__main__':
    main()
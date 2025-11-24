import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataset import SatelliteDataset
from model.u_net import UNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    dataset = SatelliteDataset(csv_file='../data/train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # model 초기화
    model = UNet().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(100):  # 100 에폭 동안 학습
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

        torch.save(model.state_dict(), f'./output/UNet/check_point_{epoch+1}.pth')
        print("Model saved to unet_model.pth")
    


if __name__ == '__main__':
    main()
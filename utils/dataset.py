import cv2
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .utils import rle_decode # utils.py 에서 rle_decode 함수 import



class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, root_dir="./data"):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_path = self.data.iloc[idx, 1]  # ./train_img/XXX.png
        clean_path = raw_path.lstrip("./") # train_img/XXX.png 로 정리

        full_path = os.path.join(self.root_dir, clean_path)
        image=cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        if pd.isna(mask_rle) or str(mask_rle).strip() == "":
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
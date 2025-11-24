import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

# =========================================
# 경로 설정
# =========================================
train_csv = './data/train.csv'
train_img_dir = './data/train_img/'
val_img_dir = './data/val_img/'
val_csv_path = './data/val.csv'

os.makedirs(val_img_dir, exist_ok=True)

# =========================================
# RLE 디코딩 / 인코딩
# =========================================
def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = np.asarray(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        img[s:e] = 1

    return img.reshape(shape)

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

# =========================================
# 1️⃣ train.csv 읽기
# =========================================
df = pd.read_csv(train_csv)
print(f"전체 train 데이터 개수: {len(df)}")

# =========================================
# 2️⃣ 무작위 500장 선택
# =========================================
df_val = df.sample(n=500, random_state=42).reset_index(drop=True)
print(f"Validation용 500장 선택 완료.")

# =========================================
# 타일링 파라미터
# =========================================
orig_size = 1024
tile = 224
crop_size = tile * 4   # 224 × 4 = 896

# =========================================
# val.csv 내용 저장 리스트
# =========================================
val_records = []

# =========================================
# 3️⃣ 타일링
# =========================================
for i, row in tqdm(df_val.iterrows(), total=len(df_val)):
    img_id = row['img_id']     # ← 스크린샷의 컬럼명과 동일하게 수정
    mask_rle = row['mask_rle']

    img_path = os.path.join(train_img_dir, f"{img_id}.png")
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ 이미지 없음: {img_path_original}")
        continue

    # 마스크 디코딩
    mask = rle_decode(mask_rle, (orig_size, orig_size))

    # 좌상단 896×896 영역만 사용
    img_crop = img[:crop_size, :crop_size]
    mask_crop = mask[:crop_size, :crop_size]

    # 총 16개 타일 생성
    count = 1
    for y in range(0, crop_size, tile):
        for x in range(0, crop_size, tile):

            img_tile = img_crop[y:y+tile, x:x+tile]
            mask_tile = mask_crop[y:y+tile, x:x+tile]

            # 타일 이름
            tile_name = f"VAL_{img_id}_{count}.png"
            tile_path = f"./val_img/{tile_name}"


            # 이미지 저장
            # 실제 이미지 저장 경로
            save_path = os.path.join(val_img_dir, tile_name)   # ./data/val_img/VAL_...
            
            # 이미지 저장
            cv2.imwrite(save_path, img_tile)

            # 마스크 다시 RLE
            mask_rle_tile = rle_encode(mask_tile)

            # 🔥 스크린샷 형태: img_id | img_path | mask_rle
            val_records.append([tile_name, tile_path, mask_rle_tile])

            count += 1

# =========================================
# 4️⃣ val.csv 저장 (스크린샷과 동일 구조)
# =========================================
val_df = pd.DataFrame(val_records, columns=['img_id', 'img_path', 'mask_rle'])
val_df.to_csv(val_csv_path, index=False)

print(f"\n🎉 Validation 타일 생성 완료!")
print(f"총 타일 수: {len(val_df)} (500장 × 16 = 8000개)")
print(f"📂 val 이미지 저장 폴더: {val_img_dir}")
print(f"📝 val.csv 저장 완료: {val_csv_path}")

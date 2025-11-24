import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- 1. 유틸리티 함수 정의 ---

def rle_decode(mask_rle, shape):
    """
    RLE 디코딩 함수
    mask_rle: RLE 인코딩 문자열
    shape: (height, width)
    """
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8) # RLE가 없는 경우 빈 마스크 반환
        
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def get_mask_pixels(mask_rle):
    """
    RLE 문자열에서 총 픽셀 수를 계산하는 함수
    """
    if pd.isna(mask_rle):
        return 0
    s = mask_rle.split()
    lengths = s[1:][::2] # '길이' 값만 추출
    if len(lengths) == 0:
        return 0
    return np.array(lengths, dtype=int).sum()

def visualize_sample(img_path, mask_rle, target_shape, save_path):
    """
    이미지와 마스크, 오버레이를 시각화하고 저장하는 함수
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: 이미지를 로드할 수 없습니다. 경로: {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = rle_decode(mask_rle, (target_shape[0], target_shape[1]))
        
        # 마스크를 3채널로 확장 (시각화를 위해)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255 # 0, 1 -> 0, 255
        
        overlay = cv2.addWeighted(img, 0.7, mask_3ch, 0.3, 0)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        ax1.imshow(img)
        ax1.set_title(f"Original Image ({img_path})")
        ax1.axis('off')
        
        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Mask")
        ax2.axis('off')
        
        ax3.imshow(overlay)
        ax3.set_title("Image with Mask Overlay")
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig) # 메모리 해제
        print(f"샘플 이미지 저장 완료: {save_path}")

    except Exception as e:
        print(f"Error in visualize_sample (path: {img_path}): {e}")

def compare_train_test_images(train_df, test_df, save_path):
    """
    학습/테스트 이미지 비교 (Resize vs Crop)
    """
    try:
        # 학습 이미지 랜덤 샘플
        train_sample = train_df.sample(1).iloc[0]
        train_img = cv2.imread(train_sample['img_path'])
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
        
        # 테스트 이미지 랜덤 샘플
        test_sample = test_df.sample(1).iloc[0]
        test_img = cv2.imread(test_sample['img_path'])
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        # 1: 학습 이미지를 224로 리사이즈
        train_img_resized = cv2.resize(train_img, (224, 224))
        
        # 2: 학습 이미지 중앙을 224로 크롭
        h, w, _ = train_img.shape
        start_h = (h - 224) // 2
        start_w = (w - 224) // 2
        train_img_cropped = train_img[start_h:start_h+224, start_w:start_w+224]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        ax1.imshow(train_img)
        ax1.set_title(f"Train (Original 1024x1024)\n{train_sample['img_id']}")
        ax1.axis('off')

        ax2.imshow(test_img)
        ax2.set_title(f"Test (Original 224x224)\n{test_sample['img_id']}")
        ax2.axis('off')
        
        ax3.imshow(train_img_resized)
        ax3.set_title("Train -> Resized (Hypothesis 1)")
        ax3.axis('off')
        
        ax4.imshow(train_img_cropped)
        ax4.set_title("Train -> Cropped (Hypothesis 2)")
        ax4.axis('off')
        
        plt.suptitle("Train vs Test Image Visual Comparison", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"학습/테스트 비교 이미지 저장 완료: {save_path}")
        
    except Exception as e:
        print(f"Error in compare_train_test_images: {e}")


# --- 2. EDA 메인 스크립트 실행 ---

def main():
    print("EDA 스크립트를 시작합니다.")
    
    # --- 단계 1: CSV 메타데이터 확인 ---
    try:
        train_df = pd.read_csv('./train.csv')
        test_df = pd.read_csv('./test.csv')
    except FileNotFoundError:
        print("Error: train.csv 또는 test.csv 파일을 찾을 수 없습니다.")
        print("현재 작업 디렉토리:", os.getcwd())
        print("디렉토리 내용:", os.listdir())
        return

    print(f"총 학습 샘플: {len(train_df)}")
    print(f"총 테스트 샘플: {len(test_df)}")
    
    print("\n--- train.csv info (Null 값 확인) ---")
    # mask_rle가 7140 non-null인지 확인
    train_df.info() 
    
    # --- 단계 2: 마스크(Mask) 분석 ---
    print("\n--- 단계 2: 마스크(Mask) 분석 중 ---")
    tqdm.pandas(desc="RLE 픽셀 수 계산")
    train_df['mask_pixels'] = train_df['mask_rle'].progress_apply(get_mask_pixels)
    
    # 1024*1024 = 1,048,576
    train_df['mask_coverage_ratio'] = train_df['mask_pixels'] / (1024 * 1024)
    
    print("\n--- 마스크 비율 통계 ---")
    print(train_df['mask_coverage_ratio'].describe())
    
    # 마스크 비율 히스토그램 시각화
    plt.figure(figsize=(12, 6))
    sns.histplot(train_df['mask_coverage_ratio'], bins=50, kde=True)
    plt.title('Mask Coverage Ratio Distribution (Train Set)')
    plt.xlabel('Mask Coverage Ratio (pixels / (1024*1024))')
    plt.ylabel('Frequency')
    plt.savefig('mask_coverage_histogram.png')
    plt.close()
    print("마스크 비율 히스토그램 저장 완료: mask_coverage_histogram.png")

    # --- 단계 3: 이미지 및 마스크 시각화 ---
    print("\n--- 단계 3: 샘플 이미지 시각화 중 ---")
    
    # 마스크가 가장 작은 샘플 (0인 경우 제외)
    min_pixels = train_df[train_df['mask_pixels'] > 0]['mask_pixels'].min()
    min_sample = train_df[train_df['mask_pixels'] == min_pixels].iloc[0]
    visualize_sample(min_sample['img_path'], min_sample['mask_rle'], (1024, 1024), 'min_coverage_sample.png')
    
    # 마스크가 가장 큰 샘플
    max_pixels = train_df['mask_pixels'].max()
    max_sample = train_df[train_df['mask_pixels'] == max_pixels].iloc[0]
    visualize_sample(max_sample['img_path'], max_sample['mask_rle'], (1024, 1024), 'max_coverage_sample.png')

    # --- 단계 4: 학습 vs 테스트 이미지 비교 ---
    print("\n--- 단계 4: 학습/테스트 이미지 비교 ---")
    compare_train_test_images(train_df, test_df, 'train_test_comparison.png')

    print("\nEDA 스크립트가 완료되었습니다.")
    print("생성된 파일: mask_coverage_histogram.png, min_coverage_sample.png, max_coverage_sample.png, train_test_comparison.png")

if __name__ == "__main__":
    main()
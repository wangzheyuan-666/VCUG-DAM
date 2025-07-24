import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# RGB color to class ID mapping
COLOR2ID = {
    (0, 0, 0): 0,         # background
    (255, 0, 0): 1,       # bladder
    (0, 255, 0): 2,       # urethra
    (0, 0, 255): 3,       # VUR
    (255, 255, 0): 4      # optional
}

def color_mask_to_label(mask_rgb):
    """
    Convert RGB mask to single-channel label mask.
    """
    mask_np = np.array(mask_rgb)
    label_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
    for color, label in COLOR2ID.items():
        match = np.all(mask_np == color, axis=-1)
        label_mask[match] = label
    return label_mask

class VCUGDataset(Dataset):
    def __init__(self, dataframe, image_dir, mask_dir, transform=None):
        """
        VCUG dataset for semantic segmentation and multi-label classification.
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']

        # Load image and mask
        image = np.array(Image.open(os.path.join(self.image_dir, filename)).convert('RGB'))
        mask_rgb = Image.open(os.path.join(self.mask_dir, filename.replace('.jpg', '.png'))).convert('RGB')
        mask = color_mask_to_label(mask_rgb)

        # Apply transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long() 
        else:
            image = ToTensorV2()(image=image)["image"]
            mask = torch.from_numpy(mask).long()

        # Multi-label classification target
        label = torch.tensor([
            row['bladder'],
            row['urethra'],
            row['l_vur'],
            row['r_vur']
        ], dtype=torch.long)

        return image, mask, label

def get_vcug_datasets(
    root_dir='',
    val_ratio=0.2,
    crop_size=256,
    random_seed=42
):
    """
    Split dataset into train/val and apply multi-scale + random crop augmentation.
    """
    df = pd.read_csv(os.path.join(root_dir, 'labels.csv'))

    # Split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_seed,
        # stratify=df[['bladder', 'urethra', 'l_vur', 'r_vur']]
    )

    image_dir = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks_vis')


    train_transform = A.Compose([
        A.RandomScale(scale_limit=(0.9, 1.1), p=0.5),  # 轻微缩放，避免严重形变
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=0),
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Rotate(limit=10, p=0.3, border_mode=0),  # 小角度旋转 ±10°
        # A.GaussianNoise(var_limit=(10.0, 50.0), p=0.3),  # 模拟成像噪声
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=5, p=0.2),  # 可选：模拟形变
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2()
    ])

    # Validation transforms
    val_transform = A.Compose([
        A.Resize(crop_size, crop_size),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = VCUGDataset(train_df, image_dir, mask_dir, transform=train_transform)
    val_dataset = VCUGDataset(val_df, image_dir, mask_dir, transform=val_transform)

    return train_dataset, val_dataset
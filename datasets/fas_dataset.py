import logging
import os
import sys
sys.path.append("./../")
import torch
from skimage import io
from torch.utils.data import Dataset
from utils.img_io import read_img, write_img
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A


class FAS_dataset(Dataset):
    def __init__(self, img_dir, is_training, file_names = '', mask_dir='', transforms=None):
 
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.is_training = is_training
        self.transform = transforms

        self.file_names = file_names

        self.diversity_transform = A.Compose([
                A.GaussNoise(p=1.0),
                A.ChannelShuffle(p=1.0),
                A.RandomBrightness(p=1.0),
                A.RandomContrast(p=1.0),
                A.Solarize(threshold=0.2, p=1.0),
                ToTensorV2(p=1.0) 
            ])

        logging.info(f'Creating dataset with {len(self.file_names)} examples')


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        file_name = self.file_names[i]
 
        img_file_path = os.path.join(self.img_dir, file_name)
        img_file = read_img(img_path=img_file_path)[:,:,[110, 70, 33]] # 取出原始高光谱数据中的RGB波段

        if self.is_training:
            mask = np.ones((img_file.shape[0], img_file.shape[1]))
            sample = self.transform(image=img_file, mask=mask)
            transformed_sample = self.diversity_transform(image=sample['image'].permute((1,2,0)).numpy().astype(np.float32))
            return sample['image'].float(), sample['mask'].float(), transformed_sample['image'].float()

        else:
            mask_file_name = file_name + "_gt"
            mask_file_path = os.path.join(self.mask_dir, mask_file_name)
            mask_file = read_img(img_path=mask_file_path)
            mask_file[mask_file==0] = 3
            mask_file[mask_file==1] = 0
            mask_file[mask_file==3] = 1

            sample = self.transform(image=img_file, mask = mask_file)

            return sample['image'], sample['mask']
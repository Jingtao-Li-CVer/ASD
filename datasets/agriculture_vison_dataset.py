import logging
import os
import sys
sys.path.append("./../")
from skimage import io
from torch.utils.data import Dataset
from utils.img_io import read_img, write_img
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A


class AgricultureVisonDataset(Dataset):
    def __init__(self, rgb_dir, nir_dir, labels_dir, filter_files_path, is_training, diversity_transform=False, transforms=None):
        self.rgb_dir = rgb_dir
        self.nir_dir = nir_dir
        self.labels_dir = labels_dir
        self.filter_files_path = filter_files_path
        self.is_training = is_training
        self.diversity_transform = diversity_transform
        self.transform = transforms
        self.png_files = np.load(self.filter_files_path)

        if self.diversity_transform:
            self.diversity_transform_operations = A.Compose([
                A.GaussNoise(p=0.7),
                A.ChannelShuffle(p=1.0),
                A.RandomBrightness(p=1.0),
                A.RandomContrast(p=1.0),
                A.Solarize(threshold=0.2, p=1.0),
                ToTensorV2(p=1.0)
            ])

        logging.info(f'Creating dataset with {len(self.png_files)} examples')

    def __len__(self):
        return len(self.png_files)

    def _get_image_label(self,i):
        png_file_name = self.png_files[i]
        jpg_file_name = png_file_name.replace(".png", ".jpg")

        rgb_file_path = os.path.join(self.rgb_dir, jpg_file_name)
        nir_file_path = os.path.join(self.nir_dir, jpg_file_name)
        label_file_path = os.path.join(self.labels_dir, png_file_name)

        rgb_file = read_img(img_path=rgb_file_path) # 1024, 1024, 4
        nir_file = read_img(img_path=nir_file_path)
        label_file = read_img(img_path=label_file_path)/255

        image_file = np.concatenate((rgb_file, np.expand_dims(nir_file, 2)), axis=2)
        return image_file, label_file

    def __getitem__(self, i):
        
        image_file, label_file = self._get_image_label(i)

        if self.is_training:
            sample = self.transform(image=image_file, mask = label_file)
            if self.diversity_transform:
                transformed_sample = self.diversity_transform_operations(image=sample['image'].permute((1,2,0)).numpy())
                return sample['image'], sample['mask'], transformed_sample['image']
            else:
                return sample['image']
        else:
            sample = self.transform(image=image_file, mask = label_file)

            if self.diversity_transform:
                transformed_sample = self.diversity_transform_operations(image=sample['image'].permute((1,2,0)).numpy())
                return sample['image'], sample['mask'], transformed_sample['image']
            else:
                return sample['image'], sample['mask']
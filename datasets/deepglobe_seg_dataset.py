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



class DeepGlobeDataset(Dataset):
    def __init__(self, img_dir, is_training, mask_dir='', normal_id='',  diversity_transform=False, background_index=6, transforms=None):
 
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.is_training = is_training
        self.diversity_transform = diversity_transform
        self.transform = transforms
        self.background_index = background_index
        self.normal_id = list(map(float, normal_id.split(', ')))

        self.files = os.listdir(self.img_dir)

        if self.diversity_transform:
            self.diversity_transform_operations = A.Compose([
                A.GaussNoise(p=1.0),
                A.ChannelShuffle(p=1.0),
                A.RandomBrightness(p=1.0),
                A.RandomContrast(p=1.0),
                A.Solarize(threshold=0.2, p=1.0),
                ToTensorV2(p=1.0) 
            ]) 

        logging.info(f'Creating dataset with {len(self.files)} examples')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file_name = self.files[i]
 
        img_file_path = os.path.join(self.img_dir, file_name)
        img_file = read_img(img_path=img_file_path)

        mask_file_name = file_name.replace('.jpg', '.png').replace('sat','mask')
        mask_file_path = os.path.join(self.mask_dir, mask_file_name)

        mask_file = read_img(img_path=mask_file_path) 

        if self.is_training:
            for id in self.normal_id:
                mask_file[mask_file == id] = 100
            mask_file[mask_file != 100] = 0
            mask_file[mask_file == 100] = 1

            sample = self.transform(image=img_file, mask = mask_file)
            if self.diversity_transform:

                transformed_sample = self.diversity_transform_operations(image=sample['image'].permute((1,2,0)).numpy())

                return sample['image'], sample['mask'], transformed_sample['image']
            else:
                return sample['image']
        else:
            background_locs = np.where(mask_file==self.background_index)
            
            for id in self.normal_id:
                mask_file[mask_file == id] = 100
            mask_file[mask_file != 100] = 0
            mask_file[mask_file == 100] = 1

            mask_file[background_locs[0], background_locs[1]] = 2
            
            sample = self.transform(image=img_file, mask = mask_file)

            return sample['image'], sample['mask']
import os
import csv
import lmdb
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from io import BytesIO
from torch.utils.data import Dataset
import torch

CLASSES = ['background','face','eye','brow','mouth','nose','ear','hair','neck+cloth']

CLASSES_19 = ['background','skin','nose','eye_g','l_eye','r_eye','l_brow','r_brow', \
    'l_ear','r_ear','mouth','u_lip','l_lip','hair','hat','ear_r','neck_l','neck','cloth']

color_map = {
            0: [0, 0, 0],
            1: [239, 234, 90],
            2: [44, 105, 154],
            3: [4, 139, 168],
            4: [13, 179, 158],
            5: [131, 227, 119],
            6: [185, 231, 105],
            7: [107, 137, 198],
            8: [241, 196, 83],
        }

def color_segmap(sample_seg):
    sample_seg = torch.argmax(sample_seg, dim=1)
    sample_mask = torch.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3), dtype=torch.float)
    for key in color_map:
        sample_mask[sample_seg==key] = torch.tensor(color_map[key], dtype=torch.float)
    sample_mask = sample_mask.permute(0,3,1,2)
    return sample_mask

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, nerf_resolution=64, dataset_name='ffhq'):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('image-length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform

        self.flip = False

        self.kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        self.kernel_4 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
        self.kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            img = Image.open(BytesIO(txn.get(f'image-{str(index).zfill(7)}'.encode('utf-8')))).convert('RGB')
            
            img = img.resize((self.resolution, self.resolution), resample=Image.HAMMING)
            thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution), resample=Image.HAMMING)
            
            mask = Image.open(BytesIO(txn.get(f'label-{str(index).zfill(7)}'.encode('utf-8')))).convert('L')
            
            if mask.size[0] != self.resolution:
                mask = mask.resize((self.resolution, self.resolution), resample=Image.NEAREST)

        img = self.transform(img)
        thumb_img = self.transform(thumb_img)

        mask_np = np.array(mask)

        cnt = torch.zeros([9, self.resolution, self.resolution])

        for idx in range(9):
            for i in range(self.resolution):
                for j in range(self.resolution):
                    if mask_np[i][j] == idx:
                        cnt[idx][i][j] = 1
            
            if idx in [2, 4, 5, 10, 11, 12]:
                cnt[idx] = torch.Tensor(cv2.dilate(np.array(cnt[idx]), self.kernel_3, 1))
            if idx in [6, 7]:
                cnt[idx] = torch.Tensor(cv2.dilate(np.array(cnt[idx]), self.kernel_4, 1))
            if idx in [8, 9]:
                cnt[idx] = torch.Tensor(cv2.dilate(np.array(cnt[idx]), self.kernel_5, 1))

        cnt = cnt.float()
        cnt = torch.clamp(cnt, 0, 1)
        
        return img, thumb_img, mask
"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np

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


class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, mask_path, img_size, label_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        self.mask = glob.glob(mask_path)
        self.data.sort()
        self.mask.sort()
        self.img_size = img_size
        self.label_size = label_size
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        assert len(self.mask) > 0, "Can't find data; make sure you specify the path to your mask"
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    def __len__(self):
        return len(self.data)
    
    def _onehot_mask(self, mask):
        label_size = self.label_size
        labels = np.zeros((label_size, mask.shape[0], mask.shape[1]))
        for i in range(label_size):
            labels[i][mask==i] = 1.0
        
        return labels

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X_hi = X.resize((256, 256), resample=PIL.Image.HAMMING)
        X_hi = self.transform(X_hi)
        X = X.resize((self.img_size, self.img_size), resample=PIL.Image.HAMMING)
        X = self.transform(X)
        MASK = PIL.Image.open(self.mask[index])
        MASK = MASK.resize((self.img_size, self.img_size), resample=PIL.Image.NEAREST)
        MASK = self._onehot_mask(np.array(MASK))
        MASK = torch.tensor(MASK, dtype=torch.float) * 2 - 1

        return X_hi, X, MASK

class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0

class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, 3

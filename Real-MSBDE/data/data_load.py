import os
import torch
import numpy as np
import cv2
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    # if use_transform:
    #     transform = PairCompose(
    #         [
    #             PairRandomCrop(96),
    #             PairRandomHorizontalFilp(),
    #             PairToTensor()
    #         ]
    #     )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'valid')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


def downscale(images):
    #print images
    ih, iw = images.shape[:2]
    downs = [[[0 for p in range(3)] for k in range(iw)] for j in range(ih)]

    for j in range(ih):
        for k in range(len(images[j])):
            for p in range(len(images[j][k])):
                tmp = bin(images[j][k][p])
                tmp_quan = tmp+'00000000'
                downs[j][k][p] = int(tmp_quan, 2)
    downs_np = np.array(downs, dtype=np.int64)
    return downs_np


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list_LBD = os.listdir(os.path.join(image_dir, '8_bit/'))
        self.image_list_HBD = os.listdir(os.path.join(image_dir, '16_bit/'))
        self._check_image(self.image_list_LBD)
        self._check_image(self.image_list_HBD)
        self.image_list_LBD.sort()
        self.image_list_HBD.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list_LBD)

    def __getitem__(self, idx):

        image_BGR = cv2.imread(os.path.join(self.image_dir, '8_bit/', self.image_list_LBD[idx]), -1)
        label_BGR = cv2.imread(os.path.join(self.image_dir, '16_bit/', self.image_list_HBD[idx]), -1)
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label_BGR, cv2.COLOR_BGR2RGB)

        image_zp = downscale(image)
        label = label.astype(np.int64)
        image_zp = image_zp/65535.0
        label = label/65535.0

        if self.transform:
            image_zp, label = self.transform(image_zp, label)
        else:
            image_zp = F.to_tensor(image_zp)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list_HBD[idx]
            return image_zp, label, name
        return image_zp, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

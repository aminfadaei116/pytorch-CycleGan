
import os 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.types import Number
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import albumentations as A


class CityscapesDataSet(Dataset):
    def __init__(self, SegmentationPath, GroundImagePath, transform=None):
        self.GroundImagePath = GroundImagePath
        self.SegmentationPath = SegmentationPath
        self.transform = transform

        self.GroundImages = os.listdir(GroundImagePath)
        self.SegmentImages = os.listdir(SegmentationPath)
        self.length_dataset = max(len(self.GroundImages), len(self.SegmentImages)) 
        self.Gr_len = len(self.GroundImages)
        self.Sg_len = len(self.SegmentImages)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        Ground = self.GroundImages[index % self.Gr_len]
        Segment = self.SegmentImages[index % self.Sg_len]


        Ground = np.array(Image.open(os.path.join(self.GroundImagePath, Ground)).convert("RGB"))
        Segment = np.array(Image.open(os.path.join(self.SegmentationPath, Segment)).convert("RGB"))

        if self.transform:
            Ground = self.transform(image=Ground)
            Segment = self.transform(image=Segment)
            Ground = Ground['image']
            Segment = Segment['image']

        return Segment, Ground

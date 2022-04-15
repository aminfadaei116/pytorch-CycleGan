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

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, down=True, use_act=True, **kwargs):
        super().__init__()
        if down:
          self.conv1 = nn.Conv2d(in_channel, out_channel, padding_mode="reflect", kernel_size=kernel_size, padding=padding, **kwargs)
        else:
          self.conv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, **kwargs)
        self.IN = nn.InstanceNorm2d(out_channel)
        if use_act:
          self.acv = nn.ReLU(inplace=True)
        else:
          self.acv = nn.Identity()
    def forward(self, x):
      x = self.conv1(x)
      x = self.IN(x)
      return  self.acv(x)
    
   
  
class ResBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.Conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1)
    self.Conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1, use_act=False)
  
  def forward(self, x):
    x2 = self.Conv1(x)
    x2 = self.Conv2(x2)
    return x + x2
  

class Generator(nn.Module):
  def __init__(self, img_channels, num_features=64, num_residual=9):
    super().__init__()
    self.conv1 = nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
    self.acv1 = nn.ReLU(inplace=True)

    self.DownBlock = nn.ModuleList([
        nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
        nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
    ])

    self.residual_blocks = nn.Sequential(
        *[ResBlock(num_features*4) for _ in range(num_residual)]
    )

    self.up_blocks = nn.ModuleList([
        ConvBlock(num_features*4, num_features*2, kernel_size=3, stride=2, padding=1, output_padding=1, down=False),
        ConvBlock(num_features*2, num_features*1, kernel_size=3, stride=2, padding=1, output_padding=1, down=False),
    ])

    self.last_layer = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

  def forward(self, x):
    x = self.conv1(x)
    x = self.acv1(x)
    for layer in self.DownBlock:
      x = layer(x)

    for layer in self.up_blocks:
      x = layer(x)
    return torch.tanh(self.last_layer(x))
  
  

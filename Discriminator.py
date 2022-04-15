
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



class DiscBlock(nn.Module):
  def __init__(self, in_channel, out_channel, stride):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channel, out_channel, 4, stride, 1, bias=True, padding_mode="reflect")
    self.IN1 = nn.InstanceNorm2d(out_channel)
    self.act1 = nn.LeakyReLU(0.2)
  def forward(self, x):
    x = self.conv1(x)
    x = self.IN1(x)
    x = self.act1(x)
    return x
  
  
class Discriminator(nn.Module):
  def __init__(self, in_channel, num_features=[64, 128, 256, 512]):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channel, num_features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect")
    self.act1 = nn.LeakyReLU(0.2)

    layers = []
    in_channel = num_features[0]
    
    for counter in range(1,len(num_features)):

      layers.append(DiscBlock(in_channel, num_features[counter], stride=1 if counter==(len(num_features)-1) else 2))
      in_channel = num_features[counter]
    layers.append(nn.Conv2d(in_channel, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
    self.model = nn.Sequential(*layers)


  def forward(self, x):
    x = self.conv1(x)
    x = self.act1(x)
    x = self.model(x)
    return torch.sigmoid(x)

   

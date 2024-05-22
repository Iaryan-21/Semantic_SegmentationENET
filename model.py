import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class enet(nn.Module):
  def __init__(self, num_classes):
    super(enet, self).__init__()
    self.initial = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.bottleneck1_0 = nn.Sequential(
        nn.Conv2d(16, 64, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.PReLU(64)
    )
    self.bottleneck1_1 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1, bias=False),
        nn.BatchNorm2d(64),
        nn.PReLU(64),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.PReLU(64),
        nn.Conv2d(64, 64, kernel_size=1, bias=False),
        nn.BatchNorm2d(64),
        nn.PReLU(64)
    )
    self.bottleneck1_2 = self.bottleneck1_1
    self.bottleneck1_3 = self.bottleneck1_1
    self.bottleneck1_4 = self.bottleneck1_1


    self.bottleneck2_0 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128)
    )
    self.bottleneck2_1 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128)
    )
    self.bottleneck2_2 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128)
    )
    self.bottleneck2_3 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=(1, 5), padding=(0, 2), bias=False),
        nn.Conv2d(128, 128, kernel_size=(5, 1), padding=(2, 0), bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128)
    )
    self.bottleneck2_4 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128)
    )
    self.bottleneck2_5 = self.bottleneck2_1
    self.bottleneck2_6 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=8, dilation=8, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128)
    )
    self.bottleneck2_7 = self.bottleneck2_3
    self.bottleneck2_8 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=16, dilation=16, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128),
        nn.Conv2d(128, 128, kernel_size=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(128)
    )
    self.bottleneck3_1 = self.bottleneck2_1
    self.bottleneck3_2 = self.bottleneck2_2
    self.bottleneck3_3 = self.bottleneck2_3
    self.bottleneck3_4 = self.bottleneck2_4
    self.bottleneck3_5 = self.bottleneck2_1
    self.bottleneck3_6 = self.bottleneck2_6
    self.bottleneck3_7 = self.bottleneck2_3
    self.bottleneck3_8 = self.bottleneck2_8


    self.bottleneck4_0 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.PReLU(64)
    )
    self.bottleneck4_1 = self.bottleneck1_1
    self.bottleneck4_2 = self.bottleneck1_1
    self.bottleneck5_0 = nn.Sequential(
        nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(16),
        nn.PReLU(16)
    )
    self.bottleneck5_1 = self.bottleneck1_1
    self.fullconv = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1)

  def forward(self, x):
    #bottleneck 1
    x = self.initial(x)
    x = self.bottleneck1_0(x)
    x = self.bottleneck1_1(x)
    x = self.bottleneck1_2(x)
    x = self.bottleneck1_3(x)
    x = self.bottleneck1_4(x)
    #bottleneck 2
    x = self.bottleneck2_0(x)
    x = self.bottleneck2_1(x)
    x = self.bottleneck2_2(x)
    x = self.bottleneck2_3(x)
    x = self.bottleneck2_4(x)
    x = self.bottleneck2_5(x)
    x = self.bottleneck2_6(x)
    x = self.bottleneck2_7(x)
    x = self.bottleneck2_8(x)
    #bottleneck 3
    x = self.bottleneck3_1(x)
    x = self.bottleneck3_2(x)
    x = self.bottleneck3_3(x)
    x = self.bottleneck3_4(x)
    x = self.bottleneck3_5(x)
    x = self.bottleneck3_6(x)
    x = self.bottleneck3_7(x)
    x = self.bottleneck3_8(x)
    #bottleneck 4
    x = self.bottleneck4_0(x)
    x = self.bottleneck4_1(x)
    x = self.bottleneck4_2(x)
    #bottleneck 5
    x = self.bottleneck5_0(x)
    x = self.bottleneck5_1(x)
    #full conv
    x = self.fullconv(x)

    return x
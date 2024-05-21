import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import os
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_transform = label_transform
       
        print(f"Root directory: {root_dir}")
        image_pattern = os.path.join(root_dir, 'leftImg8bit_trainvaltest\leftImg8bit', split, '*', '*.png')
        label_pattern = os.path.join(root_dir, 'gtFine_trainvaltest\gtFine', split, '*', '*_labelIds.png')
        
        print(f"Image pattern: {image_pattern}")
        print(f"Label pattern: {label_pattern}")

        self.images = sorted(glob.glob(image_pattern))
        self.labels = sorted(glob.glob(label_pattern))

        print(f"Number of images: {len(self.images)}")
        print(f"Number of labels: {len(self.labels)}")

        if len(self.images) != len(self.labels):
            print("Mismatch in the number of images and labels.")
            print("Sample image paths:")
            for i in range(min(5, len(self.images))):
                print(self.images[i])

            print("Sample label paths:")
            for i in range(min(5, len(self.labels))):
                print(self.labels[i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

label_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=Image.NEAREST),  
    transforms.ToTensor()
])
root_dir = 'C:\\Users\\aryan\\OneDrive\\Documents\\drug_protein_interaction'

train_dataset = CityscapesDataset(
    root_dir=root_dir,
    split='train',
    transform=image_transform,
    label_transform=label_transform
)

val_dataset = CityscapesDataset(
    root_dir=root_dir,
    split='val',
    transform=image_transform,
    label_transform=label_transform
)

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    for i in range(num_images):
        if i >= len(dataset):
            break
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()  
        label = label.squeeze().numpy()  
        
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(label, cmap='gray')
        axes[i, 1].set_title('Label')
        axes[i, 1].axis('off')
    
    plt.show()

print("Training Dataset:")
show_images(train_dataset, num_images=2)
print("Validation Dataset:")
show_images(val_dataset, num_images=2)

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

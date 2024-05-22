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
from model import enet

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_transform = label_transform
        
        print(f"Root directory: {root_dir}")
        image_pattern = os.path.join(root_dir, 'leftImg8bit_trainvaltest/leftImg8bit', split, '*', '*.png')
        label_pattern = os.path.join(root_dir, 'gtFine_trainvaltest/gtFine', split, '*', '*_labelIds.png')
        
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

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
def main():
    # print("Training Dataset:")
    # show_images(train_dataset, num_images=2)
    # print("Validation Dataset:")
    # show_images(val_dataset, num_images=2)

    device = 'cpu'
    model = enet(num_classes=21).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(1).long()  

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(1).long()  

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')

if __name__ == "__main__":
    main()

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score
import wandb

wandb.init(project="tumor-segmentation")


class PETDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        else:
            mask = np.zeros_like(image)

        mask[mask == 255.0] = 1.0

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])

train_dataset = PETDataset('./tumor-segmentation/data_training/train/patient', './tumor-segmentation/data_training/train/mask', transform)
val_dataset = PETDataset('./tumor-segmentation/data_training/val/patient', './tumor-segmentation/data_training/val/mask', transform)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

#U-Net model definition
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up3 = DoubleConv(512, 256)
        self.up2 = DoubleConv(256, 128)
        self.up1 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        x4 = self.dropout(x4)
        x = self.upconv3(x4)
        x = self.up3(torch.cat([x, x3], dim=1))

        x = self.upconv2(x)
        x = self.up2(torch.cat([x, x2], dim=1))

        x = self.upconv1(x)
        x = self.up1(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return logits

#training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    train_loss = 0.0

    #training loop code
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Rozpoczęcie walidacji
    model.eval()
    val_loss = 0.0
    acc, recall, precision = [], [], []

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            val_outputs = model(val_images)
            v_loss = criterion(val_outputs, val_masks)
            val_loss += v_loss.item()

            val_outputs = torch.sigmoid(val_outputs)
            val_outputs = val_outputs > 0.5
            val_outputs = val_outputs.float()

            acc.append(accuracy_score(val_masks.cpu().flatten(), val_outputs.cpu().flatten()))
            recall.append(recall_score(val_masks.cpu().flatten(), val_outputs.cpu().flatten()))
            precision.append(precision_score(val_masks.cpu().flatten(), val_outputs.cpu().flatten()))

    val_loss /= len(val_loader)

    # Logowanie metryk do wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": np.mean(acc),
        "recall": np.mean(recall),
        "precision": np.mean(precision)
    })

    print(f'Epoch {epoch+1}, Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {np.mean(acc)}, Recall: {np.mean(recall)}, Precision: {np.mean(precision)}')

torch.save(model.state_dict(), 'unet_pet_segmentation.pth')


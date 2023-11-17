from pathlib import Path

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import dice_score as dice
from sklearn.metrics import accuracy_score, recall_score, precision_score
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.ops import distance_box_iou_loss, focal_loss, sigmoid_focal_loss


from models.unet import UNet
from augmentor import Augmentor, ImagePreprocessor, to_grayscale


os.makedirs('unet_models', exist_ok=True)


class PETDataset(Dataset):

    def __init__(self, data_path, train_test_split = 0.8):

        self.data_path = data_path
        self.train_test_split = train_test_split

        self.train_img_path, self.train_label_path, \
        self.test_img_path, self.test_label_path = self._get_patient_datapath(data_path)

        self.augmentor = Augmentor()
        
        self.is_train = True

    def __len__(self):
        if self.is_train:
            return len(self.train_img_path)
        else:
            return len(self.test_img_path)
    
    def __call__(self, is_train=True):
        new_dataset = self.__class__(self.data_path, self.train_test_split)
        new_dataset.train_img_path = self.train_img_path
        new_dataset.train_label_path = self.train_label_path
        new_dataset.test_img_path = self.test_img_path
        new_dataset.test_label_path = self.test_label_path
        new_dataset.augmentor = self.augmentor
        new_dataset.is_train = is_train
        return new_dataset
    
    def __getitem__(self, idx):
        if self.is_train:
            img = plt.imread(self.train_img_path[idx])[:,:,:3]
            label = plt.imread(self.train_label_path[idx])[:,:,:3]


            img, label = self.augmentor(img, label)
        else:
            img = plt.imread(self.test_img_path[idx])[:,:,:3]
            label = plt.imread(self.test_label_path[idx])[:,:,:3]

        img = to_grayscale(img)
        label = to_grayscale(label)

        img = torch.tensor(img, dtype = torch.float32)[None,:]
        mask = torch.tensor(label > 0, dtype = torch.float32)[None,:]

        return img, mask

    def _get_patient_datapath(self, data_path):

        img_path = Path(data_path) / "imgs"
        label_path = Path(data_path) / "labels"

        images = list(img_path.glob("*.png"))
        labels = list(label_path.glob("*.png"))

        use_synthetic = True
        if not use_synthetic:
            images = [p for p in images if "synthetic" not in str(p)]
            labels = [p for p in labels if "synthetic" not in str(p)]

        num_images = len(images)

        train_idx = np.random.choice(num_images, int(num_images*self.train_test_split), replace = False)
        test_idx = np.setdiff1d(np.arange(num_images), train_idx)

        train_paths = [images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        test_paths = [images[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        return train_paths, train_labels, test_paths, test_labels
    
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(DEVICE)

    # VERY IMPORTANT (ensures same test train split)
    seed = 42

    train_surfix = "last_minute_2_with_synthetic"

    # Config for training
    batch_size = 4
    train_test_split = 0.9
    pretrained = None#"unet_pet_segmentation_best.pth"
    lr = 1e-3
    epochs = 100

    ### DO NOT EDIT BELOW ###

    best_dice = 0.3

    set_seed(seed)

    dataset = PETDataset('data/training_data', train_test_split=train_test_split)

    train_loader = DataLoader(dataset(is_train = True), batch_size = batch_size)
    val_loader = DataLoader(dataset(is_train = False), batch_size = batch_size)

    #training
    model = UNet(device = DEVICE)


    if pretrained is not None:
        # Loading pretrained weights
        print("from pretrained")
        model.load_state_dict(torch.load(pretrained))

    # criterion = lambda x, y: sigmoid_focal_loss(x, y, reduction='mean')
    # criterion = lambda x, y: dice_loss(x, y) + sigmoid_focal_loss(x, y, reduction='mean')
    # criterion = lambda x, y: dice_loss(x, y)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    try:
        for epoch in tqdm.trange(epochs):
            model.train()
            train_loss = 0.0

            train_pbar = tqdm.tqdm(train_loader, leave = False)

            #training loop code
            for images, masks in train_pbar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            model.eval()
            val_loss = 0.0
            die, acc, recall, precision = [], [], [], []

            with torch.no_grad():
                for val_images, val_masks in val_loader:
                    val_images = val_images.to(DEVICE)
                    val_masks = val_masks.to(DEVICE)

                    val_outputs = model(val_images)
                    v_loss = criterion(val_outputs, val_masks)
                    val_loss += v_loss.item()

                    val_outputs = torch.sigmoid(val_outputs)
                    val_outputs = val_outputs > 0.5
                    val_outputs = val_outputs.float()

                    die.append(dice(val_masks.cpu().flatten(), val_outputs.cpu().flatten()))
                    acc.append(accuracy_score(val_masks.cpu().flatten(), val_outputs.cpu().flatten()))
                    recall.append(recall_score(val_masks.cpu().flatten().int(), val_outputs.cpu().flatten().int()))

            val_loss /= len(val_loader)


            if np.mean(die) > best_dice*1.05:
                best_dice = np.mean(die)
                
                torch.save(model.state_dict(), f'unet_pet_segmentation_{train_surfix}_best_{np.round(best_dice,2)}.pth')

            print(f'Epoch {epoch+1}, Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {np.mean(acc)}, Recall: {np.mean(recall)}, Precision: {np.mean(precision)}, Dice: {np.mean(die)}')
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    
    torch.save(model.state_dict(), f'unet_models/unet_pet_segmentation_{train_surfix}.pth')

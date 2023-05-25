import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import rasterio as rio
import albumentations as A
import segmentation_models_pytorch as smp
os.chdir("..")
root = os.getcwd()


MULTICLASS_MODE: str = "multiclass"
ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['no vhange' , 'change']
ACTIVATION = None
DEVICE = 'cpu'


class MultiClassSegDataset(Dataset):

    def __init__(self, df, classes=None, transform=None):
        self.df = df
        self.classes = classes
        self.transform = transform

    def __getitem__(self, idx):

        image_name = self.df.iloc[idx, 1]
        mask_name = self.df.iloc[idx, 2]
        img = rio.open(image_name)
        image = img.read()
        image = image.transpose(1, 2, 0)
        msk = rio.open(mask_name)
        mask = msk.read()
        mask = mask.transpose(1, 2, 0)
        if (self.transform is not None):
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            image = image.float()
            mask = mask.permute(2, 0, 1)
            mask = mask.long()
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            image = image.float()
            mask = mask.long()
        return image, mask

    def __len__(self):
        return len(self.df)


# Define transforms using Albumations =======================================
val_transform = A.Compose(
    [
        A.Resize(128, 128)
    ]
)

train_transform = A.Compose(
    [
        A.Resize(128, 128),
        A.ToFloat(max_value=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.MedianBlur(blur_limit=3, always_apply=False, p=0.1)
    ]
)

# /home/alireza/OSCD/models/weights/resnet18_pretrained_unetPP/last_model.pth
best_model = torch.load(os.path.join(root , 'models/weights/resnet18_pretrained_unetPP/last_model.pth'))

# Evaluate model on validation set==============================

train = pd.read_csv(os.path.join(root ,'data/train.csv'))
val = pd.read_csv(os.path.join(root , 'data/val.csv'))

valDS = MultiClassSegDataset(val, classes=CLASSES, transform=val_transform)
trainDS = MultiClassSegDataset(train , classes=CLASSES , transform = val_transform)
model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=20,
    classes=2,
    activation=ACTIVATION,
)
model.load_state_dict(best_model['model_state'])
# Visualize images, masks, and predictions=======================================
DS = trainDS
acc =[]
for i in range(len(DS)):
    n = np.random.choice(len(DS))
    image_vis = DS[i][0].permute(1,2,0)
    image_vis = image_vis.numpy()
    image_vis = image_vis.astype('uint8')
    image_vis1 = image_vis[: , : ,[0 , 1 , 2]]
    image_vis2 = image_vis[: , : ,[12 , 11 , 10]]
    image, gt_mask = DS[i]
    gt_mask = gt_mask.squeeze()
    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = model(x_tensor)
    m = nn.Softmax(dim=1)
    pr_probs = m(pr_mask)
    pr_mask = torch.argmax(pr_probs, dim=1).squeeze(1)
    pr_mask = pr_mask.squeeze().cpu()
    fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot(141)
    plt.imshow(image_vis1)
    plt.subplot(142 , sharex = ax1 , sharey = ax1)
    plt.imshow(image_vis2)
    plt.subplot(143, sharex = ax1 , sharey = ax1)
    plt.imshow(gt_mask)
    plt.subplot(144, sharex = ax1 , sharey = ax1)
    plt.imshow(pr_mask)
    plt.show()
    # break


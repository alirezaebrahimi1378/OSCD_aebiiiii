import csv
import numpy as np
import pandas as pd
import torchvision.utils
from matplotlib import pyplot as plt
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import albumentations as A
import segmentation_models_pytorch as smp
import rasterio as rio
import statistics
from torch.utils.tensorboard import SummaryWriter

os.chdir('..')
root = os.getcwd()
weight_description = 'resnet18_pretrained_unetPP'

writer = SummaryWriter(f'models/tensorboard/{weight_description}')
model_save_path = os.path.join(root, f'models/weights/{weight_description}')
metrics_result_path = os.path.join(root, f'models/metrics/{weight_description}')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)
if not os.path.isdir(metrics_result_path):
    os.mkdir(metrics_result_path)

train = pd.read_csv(os.path.join(root, 'data/train.csv'))
val = pd.read_csv(os.path.join(root, 'data/val.csv'))

ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['change', 'no_change']
ACTIVATION = None
DEVICE = 'cuda'

avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))


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
        A.Resize(128, 128),
        A.ToFloat(max_value=256)
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

# Create the datasets   ================================================
trainDS = MultiClassSegDataset(train, classes=CLASSES, transform=train_transform)
valDS = MultiClassSegDataset(val, classes=CLASSES, transform=val_transform)
print("Number of Training Samples: " + str(len(train)) + " /Number of Validation Samples: " + str(len(val)))

# Define DataLoaders ==============================================
trainDL = torch.utils.data.DataLoader(trainDS, batch_size=8, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=0, collate_fn=None,
                                      pin_memory=False, drop_last=False, timeout=0,
                                      worker_init_fn=None)
valDL = torch.utils.data.DataLoader(valDS, batch_size=1, shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=0, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None)
# create model stracture =================================
model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=20,
    classes=2,
    activation=ACTIVATION,
)

loss = smp.losses.DiceLoss(mode="multiclass")
loss.__name__ = 'Dice_loss'

optimizer = optim.Adam(params=model.parameters(), lr=0.001)

res = input('resume training or not (True or else) : ')
if res == 'True':
    checkpoint = torch.load(os.path.join(model_save_path , 'last_model.pth'),map_location=DEVICE)
    last_epoch = checkpoint['epoch']
    last_epoch = int(last_epoch)
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    optimizer.load_state_dict(checkpoint['optim_state'])
else:
    last_epoch = 0

metrics = []
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)
val_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

tacc_file_path = os.path.join(metrics_result_path, 'train_accuracy.txt')
vacc_file_path = os.path.join(metrics_result_path, 'val_accuracy.txt')
train_loss_path = os.path.join(metrics_result_path, 'train_loss.txt')
val_loss_path = os.path.join(metrics_result_path, 'val_loss.txt')
loss_diff_path = os.path.join(metrics_result_path, 'loss_difference.txt')

train_loss = []
val_loss = []
loss_difference = []
for i in range(1 + last_epoch, 8001):
    print(f"epoch = {i}")
    ############################### train loss
    train_logs = train_epoch.run(trainDL)
    tloss = train_logs['Dice_loss']
    writer.add_scalar('train loss', tloss, global_step=i)
    with open(train_loss_path, 'a') as tl:
        tl.write("%s\n" % tloss)
    ############################### validation loss
    val_logs = val_epoch.run(valDL)
    vloss = val_logs['Dice_loss']
    writer.add_scalar('validation loss', vloss, global_step=i)
    with open(val_loss_path, 'a') as vl:
        vl.write("%s\n" % vloss)

    check_point = {
        "epoch": i,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    last_model_path = os.path.join(model_save_path, 'last_model.pth')
    torch.save(check_point, last_model_path)
    ############################### loss difference
    lossD = vloss - tloss
    with open(loss_diff_path, 'a') as ld:
        ld.write("%s\n" % lossD)
    writer.add_scalar('loss difference', lossD, global_step=i)
    print(f"train_loss = {tloss} / validation_loss = {vloss} / loss_difference = {lossD}")
    if i % 2 == 0:
        val_acc = []
        train_acc = []
        for r in range(6):
            ##############################validation accuracy
            n1 = np.random.choice(len(valDS))
            image, gt_mask = valDS[n1]
            gt_mask = gt_mask.squeeze()
            x_tensor = image.to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(x_tensor)
            m = nn.Softmax(dim=1)
            pr_probs = m(pr_mask)
            pr_mask = torch.argmax(pr_probs, dim=1).squeeze(1)
            pr_mask = pr_mask.squeeze().cpu()
            total_pixels = len(torch.flatten(gt_mask))
            val_acc.append((torch.sum(pr_mask == gt_mask).item() / total_pixels) * 100)
        for t in range(100):
            #######################################train accuracy
            n2 = np.random.choice(len(trainDS))
            image, gt_mask = trainDS[n2]
            gt_mask = gt_mask.squeeze()
            x_tensor = image.to(DEVICE).unsqueeze(0)
            pr_mask = model.predict(x_tensor)
            m = nn.Softmax(dim=1)
            pr_probs = m(pr_mask)
            pr_mask = torch.argmax(pr_probs, dim=1).squeeze(1)
            pr_mask = pr_mask.squeeze().cpu()
            total_pixels = len(torch.flatten(gt_mask))
            train_acc.append((torch.sum(pr_mask == gt_mask).item() / total_pixels) * 100)

        val_acc = statistics.mean(val_acc)
        train_acc = statistics.mean(train_acc)
        writer.add_scalar('validation accuracy', val_acc, global_step=i)
        writer.add_scalar('train accuracy', train_acc, global_step=i)
        with open(tacc_file_path, 'a') as ta:
            ta.write("%s\n" % train_acc)
        with open(vacc_file_path, 'a') as va:
            va.write("%s\n" % val_acc)

        print(f"validation accuracy >>>  {val_acc} /train accuracy >>>  {train_acc}")
        checkpoint_path = os.path.join(model_save_path, f'model_epoch{i}.pth')
        torch.save(model, checkpoint_path)

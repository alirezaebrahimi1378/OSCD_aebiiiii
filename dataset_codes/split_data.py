import os
import splitfolders
import pandas as pd

os.chdir('..')
root = os.getcwd()
input_folder = os.path.join(root , 'data/merged')
output_folder = os.path.join(root , 'data/split')
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.99, .01), group_prefix=None)


train_image_path = []
for root, dirs, files in os.walk(os.path.join(output_folder , "train/images"), topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        train_image_path.append(address)

train_mask_path = []
for root, dirs, files in os.walk(os.path.join(output_folder , "train/masks"), topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        train_mask_path.append(address)

val_image_path = []
for root, dirs, files in os.walk(os.path.join(output_folder , "val/images"), topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        val_image_path.append(address)

val_mask_path = []
for root, dirs, files in os.walk(os.path.join(output_folder , "val/masks"), topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        val_mask_path.append(address)


train = {'col1': train_image_path, 'col2': train_mask_path}
train = pd.DataFrame(data=train)

val = {'col1': val_image_path, 'col2': val_mask_path}
val = pd.DataFrame(data=val)
root = os.getcwd()
train.to_csv(os.path.join(root , 'data/train.csv'))
val.to_csv(os.path.join(root , 'data/val.csv'))
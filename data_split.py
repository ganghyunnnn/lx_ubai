import os
import numpy as np
from pathlib import Path
from shutil import copyfile
from sklearn.model_selection import train_test_split

def replace_train2valtest(dir):
    val_dir = dir.replace('train', 'val')
    test_dir = dir.replace('train', 'test')

    return val_dir, test_dir

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_files_recursive(root: str, pattern):
    p = Path(root).rglob(pattern)
    file_list = [str(x) for x in p if x.is_file()]

    return file_list

def replace_path_img2label(image_path):
    label_file = image_path.replace(img_format, label_format)
    label_file = label_file.replace('images', 'labels')
    label_name = os.path.basename(label_file)

    return label_file, label_name

def run_copy_files(path_list, mode='train'):
    for i in path_list:
        f_name = os.path.basename(i)
        print(f_name)
        label_file, label_name = replace_path_img2label(i)

        img_path = os.path.join(folder_image, f_name)
        label_path = os.path.join(folder_label, f_name)
        label_path = label_path.replace(img_format, label_format)

        if mode == 'train':
            save_img_path = os.path.join(train_image_dir, f_name)
            save_label_path = os.path.join(train_label_dir, label_name)
        elif mode == 'val':
            save_img_path = os.path.join(val_image_dir, f_name)
            save_label_path = os.path.join(val_label_dir, label_name)
        elif mode == 'test':
            save_img_path = os.path.join(test_image_dir, f_name)
            save_label_path = os.path.join(test_label_dir, label_name)

        copyfile(img_path, save_img_path)
        copyfile(label_path, save_label_path)
        
# set directory & file format
folder_image = 'data/all/images'
folder_label = 'data/all/labels'
train_image_dir = 'dataset/train/images'
train_label_dir = 'dataset/train/labels'
img_format = 'png'
label_format = 'txt'

# create train, val, test folder
val_image_dir, test_image_dir = replace_train2valtest(train_image_dir)
val_label_dir, test_label_dir = replace_train2valtest(train_label_dir)

dir_list = [train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_image_dir, test_label_dir]
for i in dir_list:
    create_dir(i)

# load file names
image_path = np.array(get_files_recursive(folder_image, '*'))

# train, val, test split
train_image_paths, x_temp = train_test_split(image_path, test_size=0.2, random_state=1)
val_image_paths, test_image_paths = train_test_split(x_temp, test_size=0.5, random_state=1)

# save train, val, test data to each folder
run_copy_files(train_image_paths, mode='train')
run_copy_files(val_image_paths, mode='val')
run_copy_files(test_image_paths, mode='test')


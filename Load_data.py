import os
import glob
import numpy as np

def combine_files(images, labels, idx):
    files = {'image': images[idx], 
             'mask': labels[idx]}
    return files

def get_files(base_dir):
    images = sorted(glob.glob(os.path.join(base_dir, r"images/*")))
    labels = sorted(glob.glob(os.path.join(base_dir, r"labels/*")))
    files = [combine_files(images, labels,idx) for idx in range(len(images))]
    return np.array(files)

def get_dataset(base_dir):
    X, y = [], []
    files = get_files(base_dir)

    for file in files:
        path = file["image"]
        X.append(np.load(path).transpose((1,2,0)))

    for file in files:
        path = file["mask"]
        y.append(np.load(path).transpose((1,2,0))) 

    return X, y


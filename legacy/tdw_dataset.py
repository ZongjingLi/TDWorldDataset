'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-16 06:37:16
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-16 06:37:19
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

local = True
dataset_dir = "/Users/melkor/Documents/datasets" if local else "datasets"

def normal_img(img):
    if len(img.shape) == 4:
        if not img.shape[1] in [1,3,4]: return img.permute(0,3,1,2)
    if len(img.shape) == 3:
        if not img.shape[0] in [1,3,4]: return img.permute(2,0,1)

def identiy_masks(img):
    return

class TDWRoomDataset(Dataset):
    def __init__(self, split = "train", resolution = (128,128), root_dir = "datasets"):
        super().__init__()
        self.split = split
        self.root_dir = root_dir + "/TDWRoom"

        img_data_path = root_dir + "/TDWRoom"+ f"/{split}/img"
        self.files = os.listdir(img_data_path)

        """ add a working resolution to adapt different scenes and parameters"""
        self.transform = transforms.Resize(resolution)
    
    def __len__(self):
        return len(self.files) // 4
    
    def __getitem__(self, idx):
        root_dir = self.root_dir
        split = self.split
        img_data_path = root_dir + f"/{self.split}/img"

        data = {}
        img = torch.tensor(plt.imread(img_data_path + f"/img_{split}_{idx}.png"))
        masks = np.load(img_data_path + f"/mask_{split}_{idx}.npy")
        #masks = torch.tensor(plt.imread(img_data_path + f"/id_{split}_{idx}.png"))
        

        data["img"] = self.transform(torch.tensor(normal_img(img)))
        data["masks"] =self.transform(torch.tensor(masks).unsqueeze(0)).squeeze(0)
        return data

if __name__ == "__main__":
    dataset = TDWRoomDataset(resolution = (256,256), root_dir = dataset_dir)
    print(len(dataset))

    loader = DataLoader(dataset, shuffle = True)
    for sample in loader:sample
    print(sample["img"].shape)
    print(sample["masks"].shape)
    plt.figure("img vs gt-masks")
    plt.subplot(121)
    plt.imshow(sample["img"][0].permute(1,2,0))
    plt.subplot(122)
    plt.imshow(sample["masks"][0])
    plt.show()
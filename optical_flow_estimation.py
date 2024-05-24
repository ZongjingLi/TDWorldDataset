'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-23 10:34:07
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-23 10:34:09
 # @ Description: This file is distributed under the MIT license.
'''

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.io import read_video
from torchvision.utils import flow_to_image

plt.rcParams["savefig.bbox"] = "tight"

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"


frame1 = torch.tensor(plt.imread("/Users/melkor/Documents/datasets/Plagueworks/train/img/img_5_1.png"))
frame2 = torch.tensor(plt.imread("/Users/melkor/Documents/datasets/Plagueworks/train/img/img_5_2.png"))



img1_batch = torch.stack([frame1,]).permute(0,3,1,2)
img2_batch = torch.stack([frame2,]).permute(0,3,1,2)


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1],
        ]
    )
    batch = transforms(batch)
    return batch

img1_batch = preprocess(img1_batch).to(device)
img2_batch = preprocess(img2_batch).to(device)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

from torchvision.models.optical_flow import raft_large

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
predicted_flows = list_of_flows[-1]
flow_imgs = flow_to_image(predicted_flows)


batch_idx = 0
# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]

fig_name = "tmp"
plt.figure("visualize optical flow")
plt.subplot(121)
plt.axis("off")
plt.imshow(img1_batch[0].permute(1,2,0))
plt.subplot(122)
plt.axis("off")
plt.imshow(flow_imgs[0].permute(1,2,0))
plt.savefig(f"outputs/{fig_name}.png")
plt.show()

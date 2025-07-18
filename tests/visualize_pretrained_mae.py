import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from nets.mae.models_mae import mae_vit_base_patch16, mae_vit_large_patch16

from utils.help_func import create_path
from utils.data_utils import img_augment
from utils.visualize import run_one_image_mae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)



# ====== script settings ======
NUM_EPOCH = 100
bhsz = 128
mask_ratio = 0.75
path_save = '../saved_models/mae_pretrain_rt0.75_E100/model_E100.pt'
idx_sample = 12


# ====== dataset ======
path_radimgnet = 'C:/GitRepos/datasets/RadImageNet/MR/brain/'
dataset = ImageFolder(root=path_radimgnet, transform=img_augment)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator()
generator.manual_seed(7)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)


train_dataloader = DataLoader(train_dataset, batch_size=bhsz, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bhsz, shuffle=False)
print('len(train_dataset):', len(train_dataset))
print('len(train_dataloader):', len(train_dataloader))


# ====== model and visualize ======
net = mae_vit_base_patch16()
net.load_state_dict(torch.load(path_save)['model_state_dict'])

tg, recon, img_masked, img_paste = run_one_image_mae(
    test_dataloader,
    idx_sample,
    net,
    device,
    mask_ratio,
)
print('img_paste.shape:', img_paste.shape)

plt.rcParams['figure.figsize'] = [24, 24]
plt.subplot(2, 2, 1)
plt.imshow(tg[0], vmin=0, vmax=1)

plt.subplot(2, 2, 2)
plt.imshow(recon[0], vmin=0, vmax=1)

plt.subplot(2, 2, 3)
plt.imshow(img_masked[0], vmin=0, vmax=1)

plt.subplot(2, 2, 4)
plt.imshow(img_paste[0], vmin=0, vmax=1)
plt.show()

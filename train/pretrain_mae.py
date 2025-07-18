import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from nets.mae.models_mae import mae_vit_base_patch16, mae_vit_large_patch16

from utils.help_func import create_path
from utils.data_utils import img_augment
from utils.training_utils import pretrain_mae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)



# ====== script settings ======
NUM_EPOCH = 100
bhsz = 128
mask_ratio = 0.75
path_save = '../saved_models/mae_pretrain_rt' + str(mask_ratio) + '_E' + str(NUM_EPOCH) + '/'
create_path(path_save)


# ====== dataset ======
path_radimgnet = 'C:/GitRepos/datasets/RadImageNet/'
dataset = ImageFolder(root=path_radimgnet, transform=img_augment)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator()
generator.manual_seed(7)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)


train_dataloader = DataLoader(train_dataset, batch_size=bhsz, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bhsz, shuffle=True)
print('len(train_dataset):', len(train_dataset))
print('len(train_dataloader):', len(train_dataloader))


# ====== model and training ======
net = mae_vit_base_patch16()
optimizer = torch.optim.AdamW(net.parameters(), lr=1.5e-4)

net = pretrain_mae(
    train_dataloader,
    optimizer,
    net,
    device,
    mask_ratio,
    path_save,
    NUM_EPOCH,
)

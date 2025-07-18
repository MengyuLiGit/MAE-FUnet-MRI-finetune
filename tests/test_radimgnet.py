import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

from utils.help_func import print_var_detail
from utils.data_utils import img_augment



path_radimgnet = 'C:/TortoiseGitRepos/datasets/rin2d/'
train_dataset = ImageFolder(root=path_radimgnet, transform=img_augment)
print(len(train_dataset))

for idx, data in enumerate(train_dataset):
    print(data)
    img, class_num = data
    print('class_num:', class_num)

    # img = img_augment(img)
    img = torch.einsum('chw->hwc', img)
    print_var_detail(img)

    plt.figure()
    plt.imshow(img)
    # plt.figure()
    # plt.imshow(img[..., 0], cmap='gray')
    plt.show()

    break

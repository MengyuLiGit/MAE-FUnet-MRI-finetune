import numpy as np
from matplotlib import pyplot as plt
import random

import torch
import torchvision.transforms as T



def img_augment(img):
    img_transform = T.Compose([
        T.PILToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ])
    img = img_transform(img)
    img = img / 255  # 0-255 to 0-1
    return img

def torch_tensor_loader(tensor, batch_size, drop_fraction=0.0):
    i = 0

    n = int(drop_fraction * len(tensor))
    if n > 0:
        tensor = tensor[n:-n]
    while i < len(tensor) - batch_size:
        i = i + batch_size
        yield tensor[i - batch_size:i]
    else:
        yield tensor[i:len(tensor)]
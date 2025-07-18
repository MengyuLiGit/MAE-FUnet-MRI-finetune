import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoderViT
from .prompt_encoder_ipt import PromptEncoder
from .image_decoder import ImageDecoder
from help_func import print_var_detail
from nets.mae.models_mae import MaskedAutoencoderViT
from timm.models.layers import DropPath, trunc_normal_

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ClassifierBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 mlp_ratio: int = 2,
                 if_res: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.if_res = if_res
        self.fc1 = nn.Linear(in_dim, in_dim // mlp_ratio)
        self.fc2 = nn.Linear(in_dim // mlp_ratio, in_dim // (mlp_ratio * 2))
        self.fc3 = nn.Linear(in_dim // (mlp_ratio * 2), out_dim)
        self.in_norm = nn.LayerNorm(in_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(self.in_norm(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _init_weights(self, m):
        # don't initialize weight if it is initialized by pretrained model!
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class mae_multi_task(nn.Module):
    def __init__(self, embed_dim, conv_feats, n_colors, conv_kernel_size, res_kernel_size, tasks,
                 image_encoder: MaskedAutoencoderViT,
                 conv=default_conv,
                 class_dim: int = 2,
                 image_size: int = 256,
                 mlp_ratio: int = 1,
                 out_channel: int = 1,
                 ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tasks = tasks
        self.embed_dim = embed_dim
        self.conv_feats = conv_feats
        self.conv_kernel_size = conv_kernel_size
        self.res_kernel_size = res_kernel_size
        self.n_colors = n_colors
        self.class_dim = class_dim
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.out_channel = out_channel
        act = nn.ReLU(True)
        self.image_encoder = image_encoder

        # level_embeddings = [nn.Embedding(1, self.embed_dim) for _ in self.tasks]

        # classify blocks
        if 'classify' in self.tasks:
            self.classify_tail = nn.Sequential(
                conv(self.n_colors, self.conv_feats, self.conv_kernel_size),
                ResBlock(conv, self.conv_feats, self.res_kernel_size, act=act),
                ResBlock(conv, self.conv_feats, self.res_kernel_size, act=act),
                nn.Conv2d(self.conv_feats, self.n_colors, kernel_size=1, stride=1)
            )

            self.classify_block = ClassifierBlock(in_dim=self.image_size * self.image_size * self.n_colors,
                                                  out_dim=self.class_dim,
                                                  mlp_ratio=self.mlp_ratio, if_res=False)


        if 'extraction' in self.tasks:
            self.extraction_block = nn.Sequential(
                conv(self.n_colors, self.conv_feats, self.conv_kernel_size),
                ResBlock(conv, self.conv_feats, self.res_kernel_size, act=act),
                ResBlock(conv, self.conv_feats, self.res_kernel_size, act=act),
                ResBlock(conv, self.conv_feats, self.res_kernel_size, act=act),
                BasicBlock(conv = conv, in_channels=self.conv_feats, out_channels=self.out_channel, kernel_size=1)
                # BasicBlock(conv = conv, in_channels=self.n_colors, out_channels=1, kernel_size=1)
            )

            # self.skip_block1 = nn.Sequential(
            #     conv(self.n_colors, self.conv_feats, self.conv_kernel_size),
            #     ResBlock(conv, self.conv_feats, self.res_kernel_size, act=act),
            #     ResBlock(conv, self.conv_feats, self.res_kernel_size, act=act),
            #     BasicBlock(conv = conv, in_channels=self.conv_feats, out_channels=self.n_colors, kernel_size=1)
            #     # BasicBlock(conv = conv, in_channels=self.n_colors, out_channels=1, kernel_size=1)
            # )




    def forward(self, image: torch.Tensor, task=None):

        # get encoded image embedding
        x, mask, ids_restore = self.image_encoder.forward_encoder(image, 0)

        # print_var_detail(mask)
        # print_var_detail(ids_restore)

        if task in self.tasks:
            if task == 'classify':
                # remove cls token
                x = x[:, 1:, :]
                x = self.image_encoder.unpatchify(x)
                classify_x = self.classify_tail(x)
                classify_x = torch.flatten(classify_x, 1)
                classify_label = self.classify_block(classify_x)
                return classify_label
            elif task == 'extraction':
                # extraction_x = self.extraction_block(x)

                # add skip connection from encoder best for now
                x_encoder = x[:, 1:, :]
                x_encoder = self.image_encoder.unpatchify(x_encoder)

                x = self.image_encoder.forward_decoder(x, ids_restore)# [N, L, p*p*3]
                x = self.image_encoder.unpatchify(x)
                extraction_x = self.extraction_block(x + x_encoder)
                return extraction_x


                # three skip connection from input image
                # x_encoder = x[:, 1:, :]
                # x_encoder = self.image_encoder.unpatchify(x_encoder)
                # x_encoder = self.skip_block1(x_encoder)
                #
                # x = self.image_encoder.forward_decoder(x, ids_restore)# [N, L, p*p*3]
                # x = self.image_encoder.unpatchify(x)
                # extraction_x = self.extraction_block(x + x_encoder + image)
                # return extraction_x

                # x_encoder = x[:, 1:, :]
                # x_encoder = self.image_encoder.unpatchify(x_encoder)
                # x_encoder = self.skip_block1(x_encoder)

                # two skip connection from input image, remove encoder skip connection, add skip block to image

                # image = self.skip_block1(image)
                # x = self.image_encoder.forward_decoder(x, ids_restore)# [N, L, p*p*3]
                # x = self.image_encoder.unpatchify(x)
                # extraction_x = self.extraction_block(x + image)
                # return extraction_x


                # # remove cls token
                # x = x[:, 1:, :]
                # x = self.image_encoder.unpatchify(x)
                # x = image + x # skip connection from input
                # extraction_x = self.extraction_block(x)
                # return extraction_x
        return None

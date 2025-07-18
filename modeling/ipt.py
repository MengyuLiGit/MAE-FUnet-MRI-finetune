# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoderViT
from .prompt_encoder_ipt import PromptEncoder
from .image_decoder import ImageDecoder
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

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

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
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



from functools import partial
class ipt(nn.Module):
    def __init__(self, n_feats, n_colors, scale, conv_kernel_size, res_kernel_size,
                 image_encoder: ImageEncoderViT,
                 prompt_encoder: PromptEncoder,
                 image_decoder: ImageDecoder,
                 conv=default_conv):
        super(ipt, self).__init__()

        self.scale_idx = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.n_feats = n_feats
        self.conv_kernel_size = conv_kernel_size
        self.res_kernel_size = res_kernel_size
        self.n_colors = n_colors
        self.scale = scale
        act = nn.ReLU(True)

        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(self.n_colors, self.n_feats, self.conv_kernel_size),
                ResBlock(conv, self.n_feats, self.res_kernel_size, act=act),
                ResBlock(conv, self.n_feats, self.res_kernel_size, act=act)
            ) for _ in self.scale
        ])

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.image_decoder = image_decoder

        # self.body = VisionTransformer(img_dim=args.patch_size, patch_dim=args.patch_dim, num_channels=n_feats,
        #                               embedding_dim=n_feats*args.patch_dim*args.patch_dim, num_heads=args.num_heads,
        #                               num_layers=args.num_layers, hidden_dim=n_feats*args.patch_dim*args.patch_dim*4,
        #                               num_queries = args.num_queries, dropout_rate=args.dropout_rate, mlp=args.no_mlp ,
        #                               pos_every=args.pos_every,no_pos=args.no_pos,no_norm=args.no_norm)
        #
        self.tail = nn.ModuleList([
            nn.Sequential(
                Upsampler(conv, s, self.n_feats, act=False),
                conv(self.n_feats, self.n_colors, self.conv_kernel_size)
            ) for s in self.scale
        ])
    def forward(self, x, params, labels):
        x = x.to(self.device)
        params = params.to(self.device)
        labels = labels.to(self.device)

        # print('self.device', self.device)
        # print('params', params.get_device())
        # print('x', x.get_device())
        # print('labels', labels.get_device())

        head_out = torch.zeros((x.shape[0], self.n_feats, x.shape[-2], x.shape[-1]),
                                      device=self.device)
        for i in range(x.shape[0]):
            label = labels[i]
            head_out[i] = self.head[label](x[i].unsqueeze(0).to(self.device)).squeeze(0)

        image_embeddings = self.image_encoder(head_out)
        sparse_embeddings= self.prompt_encoder(params = params,labels = labels)
        image_decodings, ssim_pred = self.image_decoder(image_embeddings = image_embeddings, image_pe = self.prompt_encoder.get_dense_pe(),
                                             sparse_prompt_embeddings = sparse_embeddings
            )
        # print_var_detail(image_embeddings,'image_embeddings')
        # print_var_detail(sparse_embeddings,'sparse_embeddings')
        # print_var_detail(image_decodings,'image_decodings')
        # print_var_detail(head_out,'head_out')
        # res = image_decoding
        res = image_decodings + head_out

        tail_out = torch.zeros((x.shape[0], self.n_colors, x.shape[-2], x.shape[-1]),
                                      device=self.device)
        for i in range(res.shape[0]):
            label = labels[i]
            tail_out[i] = self.tail[label](res[i].unsqueeze(0).to(self.device)).squeeze(0)

        return tail_out,ssim_pred

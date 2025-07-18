import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .common import LayerNorm2d, MLPBlock
from timm.models.layers import DropPath, trunc_normal_


class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    """

    def __init__(
            self,
            in_chans: int = 1,
            out_chans: int = 1,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
            use_attention: bool = True,
            use_res: bool = False,
    ):
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            use_attention=use_attention,
            use_res=use_res,
        )

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        # print(x.shape)
        b, c, h, w = x.shape
        x = x.view(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
            self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize
        x, mean, std = self.norm(x)

        x = self.unet(x)

        # unnormalize
        x = self.unnorm(x, mean, std)

        return x


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
class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
            use_attention: bool = False,
            use_res: bool = False,
            if_classify: bool = False,
            class_dim: int = 2,
            dim: int = 256,
            mlp_ratio: int = 1,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_attention = use_attention
        self.use_res = use_res
        self.if_classify = if_classify
        self.class_dim = class_dim
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, use_res)])
        if use_attention:
            self.down_att_layers = nn.ModuleList([AttentionBlock(chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob, use_res))
            if use_attention:
                self.down_att_layers.append(AttentionBlock(ch * 2))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob, use_res)
        if use_attention:
            self.conv_att = AttentionBlock(ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        if use_attention:
            self.up_att = nn.ModuleList()
        for _ in range(num_pool_layers):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob, use_res))
            if use_attention:
                self.up_att.append(AttentionBlock(ch))
            ch //= 2

        self.out_conv = nn.Conv2d(ch * 2, self.out_chans, kernel_size=1, stride=1)

        if if_classify:
            self.out_classifier = ClassifierBlock(in_dim=dim * dim, out_dim=self.class_dim, mlp_ratio=mlp_ratio,
                                                  if_res=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        if self.use_attention:  # use attention
            # apply down-sampling layers
            for layer, att in zip(self.down_sample_layers, self.down_att_layers):
                output = layer(output)
                output = att(output)
                stack.append(output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

            output = self.conv(output)
            output = self.conv_att(output)

            # apply up-sampling layers
            for transpose_conv, conv, att in zip(self.up_transpose_conv, self.up_conv, self.up_att):
                downsample_layer = stack.pop()
                output = transpose_conv(output)

                # reflect pad on the right/botton if needed to handle odd input dimensions
                padding = [0, 0, 0, 0]
                if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # padding right
                if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # padding bottom
                if torch.sum(torch.tensor(padding)) != 0:
                    output = F.pad(output, padding, "reflect")

                output = torch.cat([output, downsample_layer], dim=1)
                output = conv(output)
                output = att(output)
            output = self.out_conv(output)

        else:  # no attention
            # apply down-sampling layers
            for layer in self.down_sample_layers:
                output = layer(output)
                stack.append(output)
                # print('down: ', output.shape)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            output = self.conv(output)
            # print('output: ', output.shape)
            # apply up-sampling layers
            for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
                downsample_layer = stack.pop()
                output = transpose_conv(output)

                # reflect pad on the right/botton if needed to handle odd input dimensions
                padding = [0, 0, 0, 0]
                if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # padding right
                if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # padding bottom
                if torch.sum(torch.tensor(padding)) != 0:
                    output = F.pad(output, padding, "reflect")
                # print('up: ', output.shape)
                output = torch.cat([output, downsample_layer], dim=1)
                # print('output after cat: ', output.shape)
                output = conv(output)
            output = self.out_conv(output)

        if self.if_classify:
            output = torch.flatten(output, 1)
            output = self.out_classifier(output)

        # if self.if_segement:

        return output


class Unet_classify(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
            use_attention: bool = False,
            use_res: bool = False,
            if_classify: bool = False,
            class_dim: int = 2,
            dim: int = 256,
            mlp_ratio: int = 1,
            input_channels: int = 3,
            n_classes: int = 1,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_attention = use_attention
        self.use_res = use_res
        self.if_classify = if_classify
        self.class_dim = class_dim
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.input_channels = input_channels
        self.n_classes = n_classes

        # classify head
        # self.classify_input_block = ConvBlock(self.input_channels, chans, drop_prob, use_res)

        self.down_sample_layers = nn.ModuleList([ConvBlock(self.input_channels, chans, drop_prob, use_res)])
        if use_attention:
            self.down_att_layers = nn.ModuleList([AttentionBlock(chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob, use_res))
            if use_attention:
                self.down_att_layers.append(AttentionBlock(ch * 2))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob, use_res)
        if use_attention:
            self.conv_att = AttentionBlock(ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        if use_attention:
            self.up_att = nn.ModuleList()
        for _ in range(num_pool_layers):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob, use_res))
            if use_attention:
                self.up_att.append(AttentionBlock(ch))
            ch //= 2

        self.out_conv1 = ConvBlock(ch * 2, ch // 2, drop_prob, use_res)
        self.out_conv2 = ConvBlock(ch // 2, ch // 8, drop_prob, use_res)

        self.out_classifier = ClassifierBlock(in_dim=dim * dim * ch // 8, out_dim=self.class_dim, mlp_ratio=mlp_ratio,
                                              if_res=False)
        if n_classes == 1:
            self.out_rescale = nn.Sigmoid()
        elif n_classes > 1:
            self.out_rescale = nn.Softmax(dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        if self.use_attention:  # use attention
            # apply down-sampling layers
            for layer, att in zip(self.down_sample_layers, self.down_att_layers):
                output = layer(output)
                output = att(output)
                stack.append(output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

            output = self.conv(output)
            output = self.conv_att(output)

            # apply up-sampling layers
            for transpose_conv, conv, att in zip(self.up_transpose_conv, self.up_conv, self.up_att):
                downsample_layer = stack.pop()
                output = transpose_conv(output)

                # reflect pad on the right/botton if needed to handle odd input dimensions
                padding = [0, 0, 0, 0]
                if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # padding right
                if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # padding bottom
                if torch.sum(torch.tensor(padding)) != 0:
                    output = F.pad(output, padding, "reflect")

                output = torch.cat([output, downsample_layer], dim=1)
                output = conv(output)
                output = att(output)
            # output = self.out_conv(output)
            output = self.out_conv1(output)
            output = self.out_conv2(output)

        else:  # no attention
            # apply down-sampling layers
            for layer in self.down_sample_layers:
                output = layer(output)
                stack.append(output)
                # print('down: ', output.shape)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            output = self.conv(output)
            # print('output: ', output.shape)
            # apply up-sampling layers
            for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
                downsample_layer = stack.pop()
                output = transpose_conv(output)

                # reflect pad on the right/botton if needed to handle odd input dimensions
                padding = [0, 0, 0, 0]
                if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # padding right
                if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # padding bottom
                if torch.sum(torch.tensor(padding)) != 0:
                    output = F.pad(output, padding, "reflect")
                # print('up: ', output.shape)
                output = torch.cat([output, downsample_layer], dim=1)
                # print('output after cat: ', output.shape)
                output = conv(output)
            # output = self.out_conv(output)
            output = self.out_conv1(output)
            output = self.out_conv2(output)

        if self.if_classify:
            output = torch.flatten(output, 1)
            output = self.out_classifier(output)
            if self.n_classes > 0:
                output = self.out_rescale(output)

        # if self.if_segement:

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            drop_prob: float,
            use_res: bool = True, ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.use_res = use_res

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_chans),
        )

        self.layers_out = nn.Sequential(
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if self.use_res:
            return self.layers_out(self.layers(image) + self.conv1x1(image))
        else:
            return self.layers_out(self.layers(image))


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class AttentionBlock(nn.Module):
    """
    Attention block with channel and spatial-wise attention mechanism.
    """

    def __init__(self, num_ch, r=2):
        super(AttentionBlock, self).__init__()
        self.C = num_ch
        self.r = r

        self.sig = nn.Sigmoid()
        # channel attention
        self.fc_ch = nn.Sequential(nn.Linear(self.C, self.C // self.r),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.C // self.r, self.C), )
        # spatial attention
        self.conv = nn.Conv2d(self.C, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, inputs):  # [N,C,H,W]
        b, c, h, w = inputs.shape
        # spatial attention
        sa = self.conv(inputs)
        sa = self.sig(sa)
        inputs_s = sa * inputs

        # channel attention
        ca = torch.abs(inputs)
        # ca = self.pool(ca)  # [B,C,1,1]
        ca = torch.mean(ca.reshape(b, c, -1), dim=2)  # [B,C]
        ca = self.fc_ch(ca)  # [B,C]
        ca = self.sig(ca).reshape(b, c, 1, 1)  #[B,C,1,1]
        inputs_c = ca * inputs

        outputs = torch.max(inputs_s, inputs_c)
        return outputs

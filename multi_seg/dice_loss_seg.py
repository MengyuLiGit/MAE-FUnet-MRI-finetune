import torch
import torch.nn as nn
import numpy as np
from torch import nn, Tensor

import torch.nn.functional as F

def softmax_helper(x):
    return torch.softmax(x, dim=1)

def sum_tensor(input_tensor, axes, keepdim=False):
    """
    Sums a tensor over multiple axes.
    Args:
        input_tensor: the input tensor
        axes: a list or tuple of axis indices to sum over
        keepdim: whether to keep summed dimensions
    Returns:
        Summed tensor
    """
    if not isinstance(axes, (list, tuple)):
        axes = (axes,)
    axes = sorted(axes, reverse=True)  # sort descending to avoid indexing errors
    for axis in axes:
        input_tensor = input_tensor.sum(dim=axis, keepdim=keepdim)
    return input_tensor

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (B, C, H, W) or (B, C, D, H, W)
    gt must be a label map (B, 1, H, W) or (B, H, W) or one-hot (B, C, H, W)
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.shape)))

    shp_x = net_output.shape
    shp_y = gt.shape

    if len(shp_x) != len(shp_y):
        gt = gt.view((shp_y[0], 1, *shp_y[1:]))

    if all(i == j for i, j in zip(net_output.shape, gt.shape)):
        y_onehot = gt
    else:
        y_onehot = torch.zeros_like(net_output)
        y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = tp * mask
        fp = fp * mask
        fn = fn * mask
        tn = tn * mask

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)
    tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class GDL(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0,
                 square=False, square_volumes=False):
        """
        Generalized Dice Loss
        Args:
            apply_nonlin: nonlinearity to apply (e.g., softmax)
            batch_dice: if True, compute Dice over batch + class instead of per-sample
            do_bg: if False, exclude background (channel 0) from dice
            smooth: smoothing factor to avoid division by zero
            square: if True, square terms when summing tp, fp, fn
            square_volumes: if True, square volume weights
        """
        super(GDL, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.square = square
        self.square_volumes = square_volumes

    def forward(self, net_output, target, loss_mask=None):
        # net_output: (B, C, H, W)
        # target: (B, 1, H, W) or (B, H, W)
        shp_x = net_output.shape
        shp_y = target.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if len(shp_x) != len(shp_y):
            target = target.view((shp_y[0], 1, *shp_y[1:]))

        if all(i == j for i, j in zip(net_output.shape, target.shape)):
            y_onehot = target
        else:
            y_onehot = torch.zeros_like(net_output)
            y_onehot.scatter_(1, target.long(), 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        if not self.do_bg:
            net_output = net_output[:, 1:]
            y_onehot = y_onehot[:, 1:]

        tp, fp, fn, _ = get_tp_fp_fn_tn(net_output, y_onehot, axes, loss_mask, self.square)

        volumes = sum_tensor(y_onehot, axes) + 1e-6
        if self.square_volumes:
            volumes = volumes ** 2

        tp = tp / volumes
        fp = fp / volumes
        fn = fn / volumes

        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis)
        fp = fp.sum(axis)
        fn = fn.sum(axis)

        dice_score = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        #return -dice_score.mean()
        return 1-dice_score.mean()  # using 1- dice_score to avoid negative loss

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc


class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(IoULoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp + self.smooth
        denominator = tp + fp + fn + self.smooth

        iou = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return 1-iou



class MCCLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])

        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels

        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth

        mcc = nominator / denominator

        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()

        return -mcc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc



import torch
import torch.nn as nn

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1.0, weight_dice=1.0,
                 log_dice=False, ignore_label=None):
        """
        Combined SoftDice (or SoftDiceSquared) + CrossEntropy loss
        Args:
            soft_dice_kwargs: kwargs for SoftDiceLoss or SoftDiceLossSquared
            ce_kwargs: kwargs for CrossEntropyLoss
            aggregate: "sum" to sum both losses
            square_dice: use squared dice if True
            weight_ce: weight for CE loss
            weight_dice: weight for Dice loss
            log_dice: apply log to dice loss
            ignore_label: label index to ignore
        """
        super().__init__()

        self.aggregate = aggregate
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.log_dice = log_dice
        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(**soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared( **soft_dice_kwargs)

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output, target):
        """
        Args:
            net_output: logits (B, C, H, W)
            target: labels (B, H, W) or (B, 1, H, W)
        Returns:
            combined loss
        """
        if self.ignore_label is not None:
            # Mask ignored labels
            mask = (target != self.ignore_label).float()
            target = target.clone()
            target[target == self.ignore_label] = 0
        else:
            mask = None

        dice_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice > 0 else 0.0
        if self.log_dice and self.weight_dice > 0:
            dice_loss = -torch.log(-dice_loss + 1e-8)

        ce_loss = self.ce(net_output, target) if self.weight_ce > 0 else 0.0
        if self.ignore_label is not None and self.weight_ce > 0:
            ce_loss = (ce_loss * mask.squeeze(1)).sum() / mask.sum()

        if self.aggregate == "sum":
            return self.weight_dice * dice_loss + self.weight_ce * ce_loss
        else:
            raise NotImplementedError("Only 'sum' aggregation is currently supported.")

class GDL_and_CE_loss(nn.Module):
    def __init__(self, gdl_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(GDL_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = GDL(softmax_helper, **gdl_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss( **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(**soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', apply_nonlin=None):
        """
        Focal Loss for multi-class tasks
        Args:
            gamma: focusing parameter
            alpha: balancing factor (can be float or list)
            reduction: mean or sum
            apply_nonlin: optional softmax/sigmoid to apply first
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.apply_nonlin = apply_nonlin

    # def forward(self, inputs, targets):
    #     """
    #     inputs: logits (B, C, H, W)
    #     targets: labels (B, H, W) or (B, 1, H, W)
    #     """
    #     if self.apply_nonlin is not None:
    #         inputs = self.apply_nonlin(inputs)
    #
    #     if targets.ndim == inputs.ndim:
    #         targets = targets[:, 0]
    #
    #     logpt = torch.log(inputs + 1e-8)
    #     ce_loss = F.nll_loss(logpt, targets.long(), reduction='none')
    #     pt = torch.exp(-ce_loss)
    #
    #     if isinstance(self.alpha, (list, tuple)):
    #         at = torch.tensor(self.alpha, device=inputs.device)[targets.long()]
    #     else:
    #         at = self.alpha
    #
    #     focal_loss = at * (1 - pt) ** self.gamma * ce_loss
    #
    #     if self.reduction == 'mean':
    #         return focal_loss.mean()
    #     elif self.reduction == 'sum':
    #         return focal_loss.sum()
    #     else:
    #         return focal_loss
    def forward(self, inputs, targets):
        if self.apply_nonlin is not None:
            inputs = self.apply_nonlin(inputs)

        # Clamp values to avoid log(0)
        inputs = torch.clamp(inputs, min=1e-5, max=1.0 - 1e-5)

        if targets.ndim == inputs.ndim:
            targets = targets[:, 0]

        logpt = torch.log(inputs)
        ce_loss = F.nll_loss(logpt, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, (list, tuple)):
            at = torch.tensor(self.alpha, device=inputs.device)[targets.long()]
        else:
            at = self.alpha

        focal_loss = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DC_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs, aggregate="sum", weight_focal=1.0, weight_dice=1.0, log_dice=False):
        """
        Combined SoftDice + Focal Loss
        """
        super().__init__()
        self.aggregate = aggregate
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.log_dice = log_dice

        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output, target):
        dice_loss = self.dc(net_output, target) if self.weight_dice > 0 else 0.0
        if self.log_dice and self.weight_dice > 0:
            dice_loss = -torch.log(-dice_loss + 1e-8)

        focal_loss = self.focal(net_output, target) if self.weight_focal > 0 else 0.0

        if self.aggregate == "sum":
            return self.weight_dice * dice_loss + self.weight_focal * focal_loss
        else:
            raise NotImplementedError("Only 'sum' aggregation is currently supported.")


class CombinedDiceFocalCELoss(nn.Module):
    def __init__(self, dice_kwargs={}, focal_kwargs={}, ce_kwargs={}, weights=(1.0, 1.0, 1.0)):
        """
        Combined SoftDice + Focal Loss + CrossEntropy Loss
        Args:
            dice_kwargs: dict for SoftDiceLoss
            focal_kwargs: dict for FocalLoss
            ce_kwargs: dict for CrossEntropyLoss
            weights: tuple of weights (w_dice, w_focal, w_ce)
        """
        super().__init__()
        self.weights = weights  # (w_dice, w_focal, w_ce)

        self.dice_loss = SoftDiceLoss(**dice_kwargs)
        self.focal_loss = FocalLoss(**focal_kwargs)
        self.ce_loss = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output, target):
        w_dice, w_focal, w_ce = self.weights

        dice = self.dice_loss(net_output, target) if w_dice > 0 else 0.0
        focal = self.focal_loss(net_output, target) if w_focal > 0 else 0.0
        ce = self.ce_loss(net_output, target) if w_ce > 0 else 0.0

        loss = w_dice * dice + w_focal * focal + w_ce * ce
        return loss


class CombinedGDLFocalCELoss(nn.Module):
    def __init__(self, gdl_kwargs={}, focal_kwargs={}, ce_kwargs={}, weights=(1.0, 1.0, 1.0)):
        """
        Combined GeneralizedDice + Focal Loss + CrossEntropy Loss
        Args:
            gdl_kwargs: dict for GDL
            focal_kwargs: dict for FocalLoss
            ce_kwargs: dict for CrossEntropyLoss
            weights: tuple of weights (w_gdl, w_focal, w_ce)
        """
        super().__init__()
        self.weights = weights  # (w_gdl, w_focal, w_ce)

        self.gdl_loss = GDL(**gdl_kwargs)
        self.focal_loss = FocalLoss(**focal_kwargs)
        self.ce_loss = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output, target):
        w_gdl, w_focal, w_ce = self.weights

        gdl = self.gdl_loss(net_output, target) if w_gdl > 0 else 0.0
        focal = self.focal_loss(net_output, target) if w_focal > 0 else 0.0
        ce = self.ce_loss(net_output, target) if w_ce > 0 else 0.0

        loss = w_gdl * gdl + w_focal * focal + w_ce * ce
        return loss
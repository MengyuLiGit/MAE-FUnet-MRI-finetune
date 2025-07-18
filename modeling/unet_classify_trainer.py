import numpy as np

import utils.ipt_util as utility

import torch

from tqdm import tqdm
from help_func import print_var_detail
import time
from utils.evaluation_utils import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor, binaryMaskIOU
import copy
from utils.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from modeling.loss_functions.dice_loss import DC_and_Focal_loss

class Trainer:
    """
        trainer for ipt model, includes train() and test() function

        Args:
        ----------
        args : argparse
            args saved in option.py
        loader_train: dataloader
            dataloader for training, expected tuple consists of pairs
        loader_test: dataloader
            dataloader for testing, expected tuple consists of pairs
        my_model: model
            ipt model with multi-heads/tails
        my_loss: nn.modules.loss._Loss
            loss modules defined in loss._init_
        ckp: checkpoint
            checkpoint class to load pretrained model if exits
        """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (data_i, image_label) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = data_i.to(self.device)  # [batch, 1, H, W]
                    image_label = image_label.to(self.device)  # [batch, num_class]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(image_label, "image_label")

                    self.optimizer.zero_grad()
                    image_label_pred = self.model(images_input)
                    loss = self.loss(image_label_pred, image_label)
                    if torch.isnan(loss):
                        print("nan loss occur")
                        # print(filename)
                        print('image_label_pred', image_label_pred)
                        print('image_label', image_label)
                        for j in range(images_input.size(0)):
                            print_var_detail(images_input[j], 'images_input')
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()
                    if batch % 2000 == 0 or batch < 4:
                        end = time.time()
                        print('trained batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            # torch.save({
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self.PATH_CKPOINT)
            # print('MODEL SAVED at epoch: ' + str(i + 1))
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            loss_test, correct, total, accuracy = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        num_nan = 0
        correct = 0
        num_positive = 0
        correct_positive = 0
        false_positive = 0
        num_negative = 0
        correct_negative = 0
        false_negative = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            for batch, (data_i, image_label) in enumerate(self.loader_test):
                if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                    timer_data, timer_model = utility.timer(), utility.timer()
                    images_input = data_i.to(self.device)  # [pair]
                    image_label = image_label.to(self.device)  # [pair]
                    timer_data.hold()
                    timer_model.tic()
                    image_label_pred = self.model(image=images_input)
                    loss = self.loss(image_label_pred, image_label)
                    if torch.isnan(loss):
                        num_nan += 1
                    else:
                        self.running_loss_test += loss.item()

                    if batch % 1000 == 0 or batch < 4:
                        end = time.time()
                        print('tested batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()

                    # evaluation metrics
                    tg = image_label.detach()
                    pred = image_label_pred.detach()

                    _, predicted = torch.max(pred.data, 1)
                    _, label_target = torch.max(tg.data, 1)

                    total += label_target.size(0)
                    correct += (predicted == label_target).sum().item()
                else:
                    break
            if self.max_num_batch_test is None:
                self.running_loss_test /= (len(self.loader_test) - num_nan)
            else:
                self.running_loss_test /= (self.max_num_batch_test - num_nan)
            accuracy = 100. * correct / total

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| correct: ' + str(correct) + '|| total: ' + str(
                  total) + '|| accuracy: ' + str(accuracy))
        print('----------------------------------------------------------------------')
        print('----------------------------------------------------------------------')
        return self.running_loss_test, correct, total, accuracy


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class TrainerSlicer(Trainer):
    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, **kwargs):
        super().__init__(loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                         PATH_MODEL, device, max_num_batch_train=max_num_batch_train,
                         max_num_batch_test=max_num_batch_test)

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (data_i, image_label) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = data_i.to(self.device)  # [batch, 1, H, W]
                    image_label = image_label.to(self.device)  # [batch, num_class]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(image_label, "image_label")

                    self.optimizer.zero_grad()
                    image_label_pred = self.model(images_input)
                    loss = self.loss(image_label_pred, image_label)
                    if torch.isnan(loss):
                        print("nan loss occur")
                        # print(filename)
                        print('image_label_pred', image_label_pred)
                        print('image_label', image_label)
                        for j in range(images_input.size(0)):
                            print_var_detail(images_input[j], 'images_input')
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()
                    if batch % 2000 == 0 or batch < 4:
                        end = time.time()
                        print('trained batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()
                else:
                    break

            # update combined slices for dataloader
            self.loader_train.dataset.random_combined_slices = self.loader_train.dataset.get_random_combined_slices()
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            # torch.save({
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self.PATH_CKPOINT)
            # print('MODEL SAVED at epoch: ' + str(i + 1))
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')

        # test model

        if show_test:
            loss_test, correct, total, accuracy = self.test()

        return self.model


class TrainerExtraction:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (images, images_masked, images_mask_binary) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(self.device)  # [batch, 1, H, W]
                    images_masked = images_masked.to(self.device)  # [batch, num_class]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_masked, "images_masked")

                    self.optimizer.zero_grad()
                    images_masked_pred = self.model(images_input)
                    loss = self.loss(images_masked_pred, images_masked)
                    if torch.isnan(loss):
                        print("nan loss occur")
                        # print(filename)
                        print('images_masked_pred', images_masked_pred)
                        print('images_masked', images_masked)
                        for j in range(images_input.size(0)):
                            print_var_detail(images_input[j], 'images_input')
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()
                    if batch % 2000 == 0 or batch < 4:
                        end = time.time()
                        print('trained batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        nmse = 0.0
        psnr = 0.0
        ssim = 0.0
        IoU = 0.0
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            for batch, (images, images_masked, images_mask_binary) in enumerate(self.loader_test):
                if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                    timer_data, timer_model = utility.timer(), utility.timer()
                    images_input = images.to(self.device)  # [pair]
                    images_masked = images_masked.to(self.device)  # [pair]
                    timer_data.hold()
                    timer_model.tic()
                    images_masked_pred = self.model(image=images_input)
                    images_mask_binary_pred = copy.deepcopy(images_masked_pred)
                    images_mask_binary_pred[images_mask_binary_pred > 0.1] = 1
                    images_mask_binary_pred[images_mask_binary_pred <= 0.1] = 0
                    loss = self.loss(images_masked_pred, images_masked)
                    if torch.isnan(loss):
                        num_nan += 1
                    else:
                        self.running_loss_test += loss.item()

                    if batch % 1000 == 0 or batch < 4:
                        end = time.time()
                        print('tested batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()

                    # evaluation metrics
                    tg = images_masked.detach().cpu()
                    pred = images_masked_pred.detach().cpu()
                    images_mask_binary = images_mask_binary.detach().cpu()
                    images_mask_binary_pred = images_mask_binary_pred.detach().cpu()
                    # print_var_detail(tg)
                    # print_var_detail(pred)

                    # print('tg.shape:', tg.shape)
                    i_nmse = calc_nmse_tensor(tg, pred)
                    i_psnr = calc_psnr_tensor(tg, pred)
                    i_ssim = calc_ssim_tensor(tg, pred)
                    i_IoU = 0
                    num_none_ioU = 0
                    for i in range(images_mask_binary_pred.shape[0]):
                        i_IoU_i = binaryMaskIOU(images_mask_binary[i], images_mask_binary_pred[i])
                        if i_IoU_i is not None:
                            i_IoU += i_IoU_i
                        else:
                            num_none_ioU += 1
                    i_IoU = i_IoU / (images_mask_binary_pred.shape[0] - num_none_ioU)

                    nmse += i_nmse
                    psnr += i_psnr
                    ssim += i_ssim
                    IoU += i_IoU
                else:
                    break
            nmse /= len(self.loader_test)
            psnr /= len(self.loader_test)
            ssim /= len(self.loader_test)
            IoU /= len(self.loader_test)
            if self.max_num_batch_test is None:
                self.running_loss_test /= (len(self.loader_test) - num_nan)
            else:
                self.running_loss_test /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim)
              + '|| IoU: ' + str(IoU))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU


class TrainerExtractionBinary:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, n_classes=1, loss_type='dice'):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        self.n_classes = n_classes
        self.loss_type = loss_type

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (images, images_masked, images_mask_binary) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(self.device)  # [batch, 1, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(self.device)
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_masked, "images_masked")

                    self.optimizer.zero_grad()
                    images_mask_binary_pred = self.model(images_input)
                    # loss = self.loss(images_mask_binary_pred, images_mask_binary)

                    if self.loss_type == 'dice':
                        if self.n_classes == 1:
                            loss = self.loss(images_mask_binary_pred.squeeze(1), images_mask_binary.squeeze(1).float())
                            loss += dice_loss(F.sigmoid(images_mask_binary_pred.squeeze(1)), images_mask_binary.squeeze(1).float(),
                                              multiclass=False)
                        else:
                            loss = self.loss(images_mask_binary_pred, images_mask_binary.squeeze(1))
                            loss += dice_loss(
                                F.softmax(images_mask_binary_pred, dim=1).float(),
                                F.one_hot(images_mask_binary.squeeze(1), self.n_classes + 1).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    elif self.loss_type == 'dice_focal':
                        loss = self.loss(images_mask_binary_pred,
                                         images_mask_binary.squeeze(1))  # define self.loss as dice_focal loss externally
                    else:
                        loss = self.loss(images_mask_binary_pred,
                                         images_mask_binary)

                    if torch.isnan(loss):
                        print("nan loss occur")
                        # print(filename)
                        print('images_mask_binary_pred', images_mask_binary_pred)
                        print('images_masked', images_masked)
                        for j in range(images_input.size(0)):
                            print_var_detail(images_input[j], 'images_input')
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()
                    if batch % 2000 == 0 or batch < 4:
                        end = time.time()
                        print('trained batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        nmse = 0.0
        psnr = 0.0
        ssim = 0.0
        IoU = 0.0
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            for batch, (images, images_masked, images_mask_binary) in enumerate(self.loader_test):
                if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                    timer_data, timer_model = utility.timer(), utility.timer()
                    images_input = images.to(self.device)  # [pair]
                    images_masked = images_masked.to(self.device)  # [pair]
                    images_mask_binary = images_mask_binary.to(self.device)
                    timer_data.hold()
                    timer_model.tic()
                    images_mask_binary_pred = self.model(image=images_input)
                    # images_mask_binary_pred = copy.deepcopy(images_masked_pred)
                    images_mask_binary_pred[images_mask_binary_pred > 0.5] = 1
                    images_mask_binary_pred[images_mask_binary_pred <= 0.5] = 0
                    images_masked_pred = images_input * images_mask_binary_pred
                    # loss = self.loss(images_mask_binary_pred, images_mask_binary)
                    if self.loss_type == 'dice':
                        if self.n_classes == 1:
                            loss = self.loss(images_mask_binary_pred.squeeze(1), images_mask_binary.squeeze(1).float())
                            loss += dice_loss(F.sigmoid(images_mask_binary_pred.squeeze(1)), images_mask_binary.squeeze(1).float(),
                                              multiclass=False)
                        else:
                            loss = self.loss(images_mask_binary_pred, images_mask_binary.squeeze(1))
                            loss += dice_loss(
                                F.softmax(images_mask_binary_pred, dim=1).float(),
                                F.one_hot(images_mask_binary.squeeze(1), self.n_classes + 1).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    elif self.loss_type == 'dice_focal':
                        loss = self.loss(images_mask_binary_pred,
                                         images_mask_binary.squeeze(1))  # define self.loss as dice_focal loss externally
                    else:
                        loss = self.loss(images_mask_binary_pred,
                                         images_mask_binary)
                    if torch.isnan(loss):
                        num_nan += 1
                    else:
                        self.running_loss_test += loss.item()

                    if batch % 1000 == 0 or batch < 4:
                        end = time.time()
                        print('tested batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()

                    # evaluation metrics
                    tg = images_masked.detach().cpu()
                    pred = images_masked_pred.detach().cpu()
                    images_mask_binary = images_mask_binary.detach().cpu()
                    images_mask_binary_pred = images_mask_binary_pred.detach().cpu()
                    # print_var_detail(tg)
                    # print_var_detail(pred)

                    # print('tg.shape:', tg.shape)
                    i_nmse = calc_nmse_tensor(tg, pred)
                    i_psnr = calc_psnr_tensor(tg, pred)
                    i_ssim = calc_ssim_tensor(tg, pred)
                    i_IoU = 0
                    num_none_ioU = 0
                    for i in range(images_mask_binary_pred.shape[0]):
                        i_IoU_i = binaryMaskIOU(images_mask_binary[i], images_mask_binary_pred[i])
                        if i_IoU_i is not None:
                            i_IoU += i_IoU_i
                        else:
                            num_none_ioU += 1
                    i_IoU = i_IoU / (images_mask_binary_pred.shape[0] - num_none_ioU)

                    nmse += i_nmse
                    psnr += i_psnr
                    ssim += i_ssim
                    IoU += i_IoU
                else:
                    break
            nmse /= len(self.loader_test)
            psnr /= len(self.loader_test)
            ssim /= len(self.loader_test)
            IoU /= len(self.loader_test)
            if self.max_num_batch_test is None:
                self.running_loss_test /= (len(self.loader_test) - num_nan)
            else:
                self.running_loss_test /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim)
              + '|| IoU: ' + str(IoU))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU


class TrainerSegBinary:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (images, images_masked, images_mask_binary) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(self.device)  # [batch, 1, H, W]
                    images_mask_binary = images_mask_binary.to(self.device)  # [batch, num_labels, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_masked, "images_masked")

                    self.optimizer.zero_grad()
                    images_mask_binary_pred = self.model(images_input)
                    loss = self.loss(images_mask_binary_pred, images_mask_binary)
                    if torch.isnan(loss):
                        print("nan loss occur")
                        # print(filename)
                        print('images_mask_binary_pred', images_mask_binary_pred)
                        print('images_masked', images_masked)
                        for j in range(images_input.size(0)):
                            print_var_detail(images_input[j], 'images_input')
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()
                    if batch % 2000 == 0 or batch < 4:
                        end = time.time()
                        print('trained batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        nmse = 0.0
        psnr = 0.0

        if self.labels is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            ssim = torch.zeros(len(self.labels))
            IoU = np.zeros(len(self.labels))
            num_valid_ioU = np.zeros(len(self.labels))
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            for batch, (images, images_masked, images_mask_binary) in enumerate(self.loader_test):
                if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                    timer_data, timer_model = utility.timer(), utility.timer()
                    images_input = images.to(self.device)  # [pair]
                    images_masked = images_masked.to(self.device)  # [pair]
                    images_mask_binary = images_mask_binary.to(self.device)
                    timer_data.hold()
                    timer_model.tic()
                    images_mask_binary_pred = self.model(image=images_input)

                    # entropy
                    # images_mask_binary_pred[images_mask_binary_pred > 0.5] = 1
                    # images_mask_binary_pred[images_mask_binary_pred <= 0.5] = 0
                    # L1
                    images_mask_binary_pred[images_mask_binary_pred > 0.1] = 1
                    images_mask_binary_pred[images_mask_binary_pred <= 0.1] = 0
                    images_masked_pred = images_input * images_mask_binary_pred
                    loss = self.loss(images_mask_binary_pred, images_mask_binary)
                    if torch.isnan(loss):
                        num_nan += 1
                    else:
                        self.running_loss_test += loss.item()

                    if batch % 1000 == 0 or batch < 4:
                        end = time.time()
                        print('tested batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()

                    # evaluation metrics
                    tg = images_masked.detach().cpu()
                    pred = images_masked_pred.detach().cpu()
                    images_mask_binary = images_mask_binary.detach().cpu()
                    images_mask_binary_pred = images_mask_binary_pred.detach().cpu()

                    i_nmse = calc_nmse_tensor(tg, pred)
                    i_psnr = calc_psnr_tensor(tg, pred)
                    i_ssim = torch.zeros(images_mask_binary_pred.shape[1])
                    i_IoU = np.zeros(images_mask_binary_pred.shape[1])
                    for j in range(images_mask_binary_pred.shape[1]):
                        i_num_none_ioU = 0
                        for i in range(images_mask_binary_pred.shape[0]):
                            i_IoU_i = binaryMaskIOU(images_mask_binary[i][j], images_mask_binary_pred[i][j])
                            i_ssim_i = calc_ssim_tensor(tg[i, j, :, :].unsqueeze(0).unsqueeze(0),
                                                        pred[i, j, :, :].unsqueeze(0).unsqueeze(0))
                            if i_IoU_i is not None:
                                i_IoU[j] += i_IoU_i
                                i_ssim[j] += i_ssim_i
                            else:
                                i_num_none_ioU += 1
                        if i_num_none_ioU < images_mask_binary_pred.shape[0]:
                            num_valid_ioU[j] += (images_mask_binary_pred.shape[0] - i_num_none_ioU)
                        else:
                            num_valid_ioU[j] += 0

                    nmse += i_nmse
                    psnr += i_psnr

                    ssim += i_ssim
                    IoU += i_IoU
                else:
                    break
            nmse /= len(self.loader_test)
            psnr /= len(self.loader_test)

            for j in range(len(self.labels)):
                num_valid = num_valid_ioU[j]
                ssim[j] /= num_valid
                IoU[j] /= num_valid
            if self.max_num_batch_test is None:
                self.running_loss_test /= (len(self.loader_test) - num_nan)
            else:
                self.running_loss_test /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim)
              + '|| IoU: ' + str(IoU))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU


class TrainerSegBinaryDiceLoss:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, scheduler, grad_scaler, amp,
                 gradient_clipping, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None, loss_type='dice'):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        self.loss_type = loss_type
        self.n_classes = len(labels)
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.gradient_clipping = gradient_clipping
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (images, images_masked, images_mask_binary) in enumerate(self.loader_train):

                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(device=self.device, dtype=torch.float32,
                                             memory_format=torch.channels_last)  # [batch, 1, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)  # [batch, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_masked, "images_masked")

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        images_mask_binary_pred = self.model(images_input)  # [batch, n_classes+1, H, W]
                        # if self.n_classes == 1: # for binary class, use dice loss only
                        if images_mask_binary_pred.shape[1] == 1:
                            loss = self.loss(images_mask_binary_pred.squeeze(1), images_mask_binary.float())
                            loss += dice_loss(F.sigmoid(images_mask_binary_pred.squeeze(1)), images_mask_binary.float(),
                                              multiclass=False)
                        else:
                            if self.loss_type == 'dice':
                                loss = self.loss(images_mask_binary_pred, images_mask_binary)
                                loss += dice_loss(
                                    F.softmax(images_mask_binary_pred, dim=1).float(),
                                    F.one_hot(images_mask_binary, self.n_classes + 1).permute(0, 3, 1, 2).float(),
                                    multiclass=True
                                )
                            elif self.loss_type == 'dice_focal':
                               # print_var_detail(images_mask_binary_pred, "images_mask_binary_pred")
                               # print_var_detail(images_mask_binary, "images_mask_binary")
                                loss = self.loss(images_mask_binary_pred, images_mask_binary) # define self.loss as dice_focal loss externally
                            else:
                                loss = self.loss(images_mask_binary_pred,
                                                 images_mask_binary)
                        self.optimizer.zero_grad()
                        if torch.isnan(loss):
                            print("nan loss occur")
                            # print(filename)
                            print('images_mask_binary_pred', images_mask_binary_pred)
                            print('images_masked', images_masked)
                            for j in range(images_input.size(0)):
                                print_var_detail(images_input[j], 'images_input')
                            num_nan += 1
                        else:
                            self.grad_scaler.scale(loss).backward()
                            self.grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                            self.grad_scaler.step(self.optimizer)
                            self.running_loss_train += loss.item()
                            self.grad_scaler.update()
                        if batch % 2000 == 0 or batch < 4:
                            end = time.time()
                            print('trained batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU, dice_score = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        dice_score = 0
        nmse = 0.0
        psnr = 0.0

        if self.labels is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            ssim = torch.zeros(len(self.labels))
            IoU = np.zeros(len(self.labels))
            num_valid_ioU = np.zeros(len(self.labels))
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                for batch, (images, _, images_mask_binary) in enumerate(self.loader_test):
                    if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                        timer_data, timer_model = utility.timer(), utility.timer()
                        images_input = images.to(device=self.device, dtype=torch.float32,
                                                 memory_format=torch.channels_last)  # [pair]
                        # images_masked = images_masked.to(self.device)  # [pair]
                        images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)
                        timer_data.hold()
                        timer_model.tic()
                        images_mask_binary_pred = self.model(image=images_input)

                        # entropy
                        # images_mask_binary_pred[images_mask_binary_pred > 0.5] = 1
                        # images_mask_binary_pred[images_mask_binary_pred <= 0.5] = 0
                        # L1
                        # images_mask_binary_pred[images_mask_binary_pred > 0.1] = 1
                        # images_mask_binary_pred[images_mask_binary_pred <= 0.1] = 0

                        # loss = self.loss(images_mask_binary_pred, images_mask_binary)
                        # if self.n_classes == 1:
                        if images_mask_binary_pred.shape[1] == 1:
                            loss = self.loss(images_mask_binary_pred.squeeze(1), images_mask_binary.float())
                            loss += dice_loss(F.sigmoid(images_mask_binary_pred.squeeze(1)), images_mask_binary.float(),
                                              multiclass=False)
                        else:
                            loss = self.loss(images_mask_binary_pred, images_mask_binary)
                            loss += dice_loss(
                                F.softmax(images_mask_binary_pred, dim=1).float(),
                                F.one_hot(images_mask_binary, self.n_classes + 1).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                        # if self.n_classes == 1:
                        if images_mask_binary_pred.shape[1] == 1:
                            assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= 1, 'True mask indices should be in [0, 1]'
                            images_mask_binary_pred = (F.sigmoid(images_mask_binary_pred) > 0.5).float()
                            # compute the Dice score
                            dice_score += dice_coeff(images_mask_binary_pred, images_mask_binary,
                                                     reduce_batch_first=False)
                        else:
                            assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= self.n_classes + 1, 'True mask indices should be in [0, n_classes['
                            # convert to one-hot format
                            images_mask_binary = F.one_hot(images_mask_binary, self.n_classes + 1).permute(0, 3, 1,
                                                                                                           2).float()
                            images_mask_binary_pred = F.one_hot(images_mask_binary_pred.argmax(dim=1),
                                                                self.n_classes + 1).permute(0, 3, 1, 2).float()
                            # compute the Dice score, ignoring background
                            dice_score += multiclass_dice_coeff(images_mask_binary_pred[:, 1:],
                                                                images_mask_binary[:, 1:],
                                                                reduce_batch_first=False)

                        if torch.isnan(loss):
                            num_nan += 1
                            print('nan loss detected')
                            continue
                        else:
                            self.running_loss_test += loss.item()

                        if batch % 1000 == 0 or batch < 4:
                            end = time.time()
                            print('tested batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()

                        # evaluation metrics

                        # if self.n_classes == 1:
                        if images_mask_binary_pred.shape[1] == 1:
                            images_masked_pred = images_input * images_mask_binary_pred
                            pred = images_masked_pred.detach().cpu()
                            images_masked = images_input * images_mask_binary
                            tg = images_masked.detach().cpu()
                            images_mask_binary = images_mask_binary.detach().cpu()
                            images_mask_binary_pred = images_mask_binary_pred.detach().cpu()
                        else:  # remove the none class from the first channel of dim 1
                            images_masked_pred = images_input * images_mask_binary_pred[:, 1:]
                            pred = images_masked_pred.detach().cpu()
                            images_masked = images_input * images_mask_binary[:, 1:]
                            tg = images_masked.detach().cpu()
                            images_mask_binary = images_mask_binary[:, 1:].detach().cpu()
                            images_mask_binary_pred = images_mask_binary_pred[:, 1:].detach().cpu()
                        # print_var_detail(tg)
                        # print_var_detail(pred)

                        # print('tg.shape:', tg.shape)
                        # print('pred.shape:', pred.shape)
                        # print('images_input.shape:', images_input.shape)
                        # print('images_mask_binary.shape:', images_mask_binary.shape)
                        # print('images_mask_binary_pred.shape:', images_mask_binary_pred.shape)
                        i_nmse = calc_nmse_tensor(tg, pred)
                        i_psnr = calc_psnr_tensor(tg, pred)
                        # i_ssim = calc_ssim_tensor(tg, pred)
                        i_ssim = torch.zeros(images_mask_binary_pred.shape[1])
                        i_IoU = np.zeros(images_mask_binary_pred.shape[1])
                        for j in range(images_mask_binary_pred.shape[1]):
                            i_num_none_ioU = 0
                            # i_ssim[j] = calc_ssim_tensor(tg[:,j,:,:].unsqueeze(1), pred[:,j,:,:].unsqueeze(1))
                            for i in range(images_mask_binary_pred.shape[0]):
                                i_IoU_i = binaryMaskIOU(images_mask_binary[i][j], images_mask_binary_pred[i][j])
                                i_ssim_i = calc_ssim_tensor(tg[i, j, :, :].unsqueeze(0).unsqueeze(0),
                                                            pred[i, j, :, :].unsqueeze(0).unsqueeze(0))
                                if i_IoU_i is not None:
                                    i_IoU[j] += i_IoU_i
                                    i_ssim[j] += i_ssim_i
                                else:
                                    i_num_none_ioU += 1
                            if i_num_none_ioU < images_mask_binary_pred.shape[0]:
                                # i_IoU[j] = i_IoU[j] / (images_mask_binary_pred.shape[0] - i_num_none_ioU) # count it later
                                # i_ssim[j] = i_ssim[j] / (images_mask_binary_pred.shape[0] - i_num_none_ioU) # count it later
                                num_valid_ioU[j] += (images_mask_binary_pred.shape[0] - i_num_none_ioU)
                            else:
                                num_valid_ioU[j] += 0

                        nmse += i_nmse
                        psnr += i_psnr

                        ssim += i_ssim
                        IoU += i_IoU
                    else:
                        break

            for j in range(len(self.labels)):
                num_valid = num_valid_ioU[j]
                ssim[j] /= num_valid
                IoU[j] /= num_valid

            if self.max_num_batch_test is None:
                nmse /= (len(self.loader_test) - num_nan)
                psnr /= (len(self.loader_test) - num_nan)
                self.running_loss_test /= (len(self.loader_test) - num_nan)
                dice_score /= (len(self.loader_test) - num_nan)
            else:
                nmse /= (self.max_num_batch_test - num_nan)
                psnr /= (self.max_num_batch_test - num_nan)
                self.running_loss_test /= (self.max_num_batch_test - num_nan)
                dice_score /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim)
              + '|| IoU: ' + str(IoU) + '|| Dice score: ' + str(dice_score))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU, dice_score


class TrainerSegBinaryDiceLoss3D:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, scheduler, grad_scaler, amp,
                 gradient_clipping, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None, loss_type='dice', target_channel=40):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        self.loss_type = loss_type
        self.n_classes = len(labels)
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.gradient_clipping = gradient_clipping
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        if self.n_classes == 1:
            print("binary mask")
        self.target_channel = target_channel

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (images, image_label, patient_ID, direction, image_tensor_mask_total, images_mask_binary, use_argment) in enumerate(self.loader_train):

                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(device=self.device, dtype=torch.float32,
                                             memory_format=torch.channels_last)  # [batch, C, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)  # [batch, C, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_mask_binary, "images_mask_binary")

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        images_mask_binary_pred = self.model(images_input)  # [batch, C, H, W]
                        if self.n_classes == 1: # for binary class, use dice loss only
                        # if images_mask_binary_pred.shape[1] == 1:
                            if self.loss_type == 'dice':
                                for j in range(images_mask_binary_pred.shape[1]):
                                    if j == 0:
                                        loss = self.loss(images_mask_binary_pred[:, j, :, :],
                                                          images_mask_binary[:, j, :, :].float())
                                        loss += dice_loss(F.sigmoid(images_mask_binary_pred[:, j, :, :]),
                                                          images_mask_binary[:, j, :, :].float(),
                                                          multiclass=False)
                                    else:
                                        loss += self.loss(images_mask_binary_pred[:,j,:,:], images_mask_binary[:,j,:,:].float())
                                        loss += dice_loss(F.sigmoid(images_mask_binary_pred[:,j,:,:]), images_mask_binary[:,j,:,:].float(),
                                                          multiclass=False)
                            # elif self.loss_type == 'dice_focal':
                            #     for j in range(images_mask_binary_pred.shape[1]):
                            #         if j == 0:
                            #             loss = self.loss(images_mask_binary_pred[:, j, :, :],
                            #                               images_mask_binary[:, j, :, :].float())
                            #         else:
                            #             loss += self.loss(images_mask_binary_pred[:, j, :, :],
                            #                              images_mask_binary[:, j, :, :].float())

                        # else:
                        #     if self.loss_type == 'dice':
                        #         loss = self.loss(images_mask_binary_pred, images_mask_binary)
                        #         loss += dice_loss(
                        #             F.softmax(images_mask_binary_pred, dim=1).float(),
                        #             F.one_hot(images_mask_binary, self.n_classes + 1).permute(0, 3, 1, 2).float(),
                        #             multiclass=True
                        #         )
                        #     elif self.loss_type == 'dice_focal':
                        #         loss = self.loss(images_mask_binary_pred, images_mask_binary) # define self.loss as dice_focal loss externally
                        #     else:
                        #         loss = self.loss(images_mask_binary_pred,
                        #                          images_mask_binary)
                        self.optimizer.zero_grad()
                        if torch.isnan(loss):
                            print("nan loss occur")
                            # print(filename)
                            print('images_mask_binary_pred', images_mask_binary_pred)
                            for j in range(images_input.size(0)):
                                print_var_detail(images_input[j], 'images_input')
                            num_nan += 1
                        else:
                            self.grad_scaler.scale(loss).backward()
                            self.grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                            self.grad_scaler.step(self.optimizer)
                            self.running_loss_train += loss.item()
                            self.grad_scaler.update()
                        if batch % 100 == 0 or batch < 4:
                            end = time.time()
                            print('trained batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU, dice_score = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        dice_score = 0
        nmse = 0.0
        psnr = 0.0

        if self.target_channel is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            # ssim = torch.zeros(len(self.labels))
            # IoU = np.zeros(len(self.labels))
            # num_valid_ioU = np.zeros(len(self.labels))
            ssim = torch.zeros(self.target_channel)
            IoU = np.zeros(self.target_channel)
            num_valid_ioU = np.zeros(self.target_channel)
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                for batch, (images, image_label, patient_ID, direction, image_tensor_mask_total, images_mask_binary, use_argmenty) in enumerate(self.loader_test):
                    if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                        timer_data, timer_model = utility.timer(), utility.timer()
                        images_input = images.to(device=self.device, dtype=torch.float32,
                                                 memory_format=torch.channels_last)  # [pair]
                        # images_masked = images_masked.to(self.device)  # [pair]
                        images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)
                        timer_data.hold()
                        timer_model.tic()
                        images_mask_binary_pred = self.model(image=images_input)

                        # entropy
                        # images_mask_binary_pred[images_mask_binary_pred > 0.5] = 1
                        # images_mask_binary_pred[images_mask_binary_pred <= 0.5] = 0
                        # L1
                        # images_mask_binary_pred[images_mask_binary_pred > 0.1] = 1
                        # images_mask_binary_pred[images_mask_binary_pred <= 0.1] = 0

                        # loss = self.loss(images_mask_binary_pred, images_mask_binary)
                        if self.n_classes == 1:
                        # if images_mask_binary_pred.shape[1] == 1:
                            if self.loss_type == 'dice':
                                for j in range(images_mask_binary_pred.shape[1]):
                                    if j == 0:
                                        loss = self.loss(images_mask_binary_pred[:, j, :, :],
                                                         images_mask_binary[:, j, :, :].float())
                                        loss += dice_loss(F.sigmoid(images_mask_binary_pred[:, j, :, :]),
                                                          images_mask_binary[:, j, :, :].float(),
                                                          multiclass=False)
                                    else:
                                        loss += self.loss(images_mask_binary_pred[:, j, :, :],
                                                          images_mask_binary[:, j, :, :].float())
                                        loss += dice_loss(F.sigmoid(images_mask_binary_pred[:, j, :, :]),
                                                          images_mask_binary[:, j, :, :].float(),
                                                          multiclass=False)
                            # elif self.loss_type == 'dice_focal':
                            #     for j in range(images_mask_binary_pred.shape[1]):
                            #         if j == 0:
                            #             loss = self.loss(images_mask_binary_pred[:, j, :, :],
                            #                              images_mask_binary[:, j, :, :].float())
                            #         else:
                            #             loss += self.loss(images_mask_binary_pred[:, j, :, :],
                            #                               images_mask_binary[:, j, :, :].float())
                        # else:
                        #     loss = self.loss(images_mask_binary_pred, images_mask_binary)
                        #     loss += dice_loss(
                        #         F.softmax(images_mask_binary_pred, dim=1).float(),
                        #         F.one_hot(images_mask_binary, self.n_classes + 1).permute(0, 3, 1, 2).float(),
                        #         multiclass=True
                        #     )

                        if self.n_classes == 1:
                        # if images_mask_binary_pred.shape[1] == 1:
                            assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= 1, 'True mask indices should be in [0, 1]'
                            for k in range(images_mask_binary_pred.shape[1]):
                                # loss += self.loss(images_mask_binary_pred[:, j, :, :],
                                #                   images_mask_binary[:, j, :, :].float())
                                # loss += dice_loss(F.sigmoid(images_mask_binary_pred[:, j, :, :]),
                                #                   images_mask_binary[:, j, :, :].float(),
                                #                   multiclass=False)
                                images_mask_binary_pred[:, k, :, :] = (F.sigmoid(images_mask_binary_pred[:, k, :, :]) > 0.5).float()
                                # compute the Dice score
                                dice_score += dice_coeff(images_mask_binary_pred[:, k, :, :], images_mask_binary[:, k, :, :],
                                                         reduce_batch_first=False)
                        # else:
                        #     assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= self.n_classes + 1, 'True mask indices should be in [0, n_classes['
                        #     # convert to one-hot format
                        #     images_mask_binary = F.one_hot(images_mask_binary, self.n_classes + 1).permute(0, 3, 1,
                        #                                                                                    2).float()
                        #     images_mask_binary_pred = F.one_hot(images_mask_binary_pred.argmax(dim=1),
                        #                                         self.n_classes + 1).permute(0, 3, 1, 2).float()
                        #     # compute the Dice score, ignoring background
                        #     dice_score += multiclass_dice_coeff(images_mask_binary_pred[:, 1:],
                        #                                         images_mask_binary[:, 1:],
                        #                                         reduce_batch_first=False)

                        if torch.isnan(loss):
                            num_nan += 1
                            print('nan loss detected')
                            continue
                        else:
                            self.running_loss_test += loss.item()

                        if batch % 20 == 0 or batch < 4:
                            end = time.time()
                            print('tested batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()

                        # evaluation metrics

                        if self.n_classes == 1:
                        # if images_mask_binary_pred.shape[1] == 1:
                            images_masked_pred = images_input * images_mask_binary_pred
                            pred = images_masked_pred.detach().cpu()
                            images_masked = images_input * images_mask_binary
                            tg = images_masked.detach().cpu()
                            images_mask_binary = images_mask_binary.detach().cpu()
                            images_mask_binary_pred = images_mask_binary_pred.detach().cpu()
                        # else:  # remove the none class from the first channel of dim 1
                        #     images_masked_pred = images_input * images_mask_binary_pred[:, 1:]
                        #     pred = images_masked_pred.detach().cpu()
                        #     images_masked = images_input * images_mask_binary[:, 1:]
                        #     tg = images_masked.detach().cpu()
                        #     images_mask_binary = images_mask_binary[:, 1:].detach().cpu()
                        #     images_mask_binary_pred = images_mask_binary_pred[:, 1:].detach().cpu()
                        # print_var_detail(tg)
                        # print_var_detail(pred)

                        # print('tg.shape:', tg.shape)
                        # print('pred.shape:', pred.shape)
                        # print('images_input.shape:', images_input.shape)
                        # print('images_mask_binary.shape:', images_mask_binary.shape)
                        # print('images_mask_binary_pred.shape:', images_mask_binary_pred.shape)
                        i_nmse = calc_nmse_tensor(tg, pred)
                        i_psnr = calc_psnr_tensor(tg, pred)
                        # i_ssim = calc_ssim_tensor(tg, pred)
                        i_ssim = torch.zeros(images_mask_binary_pred.shape[1])
                        i_IoU = np.zeros(images_mask_binary_pred.shape[1])
                        for j in range(images_mask_binary_pred.shape[1]):
                            i_num_none_ioU = 0
                            # i_ssim[j] = calc_ssim_tensor(tg[:,j,:,:].unsqueeze(1), pred[:,j,:,:].unsqueeze(1))
                            for i in range(images_mask_binary_pred.shape[0]):
                                i_IoU_i = binaryMaskIOU(images_mask_binary[i][j], images_mask_binary_pred[i][j])
                                i_ssim_i = calc_ssim_tensor(tg[i, j, :, :].unsqueeze(0).unsqueeze(0),
                                                            pred[i, j, :, :].unsqueeze(0).unsqueeze(0))
                                if i_IoU_i is not None:
                                    i_IoU[j] += i_IoU_i
                                    i_ssim[j] += i_ssim_i
                                else:
                                    i_num_none_ioU += 1
                            if i_num_none_ioU < images_mask_binary_pred.shape[0]:
                                # i_IoU[j] = i_IoU[j] / (images_mask_binary_pred.shape[0] - i_num_none_ioU) # count it later
                                # i_ssim[j] = i_ssim[j] / (images_mask_binary_pred.shape[0] - i_num_none_ioU) # count it later
                                num_valid_ioU[j] += (images_mask_binary_pred.shape[0] - i_num_none_ioU)
                            else:
                                num_valid_ioU[j] += 0

                        nmse += i_nmse
                        psnr += i_psnr

                        ssim += i_ssim
                        IoU += i_IoU
                    else:
                        break

            for j in range(self.target_channel):
                num_valid = num_valid_ioU[j]
                ssim[j] /= num_valid
                IoU[j] /= num_valid
            ssim_ave_channel = sum(ssim)/len(ssim)
            IoU_ave_channel = sum(IoU) / len(IoU)

            if self.max_num_batch_test is None:
                nmse /= (len(self.loader_test) - num_nan)
                psnr /= (len(self.loader_test) - num_nan)
                self.running_loss_test /= (len(self.loader_test) - num_nan)
                dice_score /= (len(self.loader_test) - num_nan)
            else:
                nmse /= (self.max_num_batch_test - num_nan)
                psnr /= (self.max_num_batch_test - num_nan)
                self.running_loss_test /= (self.max_num_batch_test - num_nan)
                dice_score /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim) + '|| ssim_ave_channel: ' + str(ssim_ave_channel)
              + '|| IoU: ' + str(IoU)+ '|| IoU_ave_channel: ' + str(IoU_ave_channel) + '|| Dice score: ' + str(dice_score))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU, dice_score
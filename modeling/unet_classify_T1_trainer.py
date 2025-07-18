import utils.ipt_util as utility
from utils.general_utils import construct_tensor_num_to_class

import torch

from tqdm import tqdm
from help_func import print_var_detail


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

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, resume_epoch,
                 PATH_MODEL, PATH_CKPOINT, device, load_ckp: bool):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.PATH_CKPOINT = PATH_CKPOINT
        self.cpu = False
        if load_ckp:
            checkpoint = torch.load(self.PATH_CKPOINT)
            # self.optimizer.load(model_load_path, epoch=model_resume_epoch)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0


    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # training iteration
        pbar = tqdm(range(epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0
            for batch, (data_i, image_sequence, image_label) in enumerate(self.loader_train):
                image_label = construct_tensor_num_to_class(image_sequence, num_class=2)
                images_input = data_i.to(self.device)  # [batch, 1, H, W]
                image_label = image_label.to(self.device)  # [batch, 2]
                if i == 0 and batch == 0:
                    print_var_detail(images_input, "images_input")
                    print_var_detail(image_label, "image_label")
                    timer_data.hold()
                    timer_model.tic()

                if batch % 3000 == 0:
                    print("batch: ", batch)

                self.optimizer.zero_grad()

                # sr, decoder_output = self.model(x = images[0],params=params,labels=labels)
                # sr = utility.quantize(sr, self.args.rgb_range)

                # loss = self.loss(sr, images[1]) + contrastive_loss(decoder_output, weights=0.1)
                image_label_pred = self.model(images_input)

                # ssim loss
                # ssim_target = torch.zeros(ssim_pred.shape, device=self.device)
                # for i in range(sr.shape[0]):
                #     # nmse_target[i] = calc_nmse_tensor(images_target[i], sr[i])
                #     # psnr_target[i] = calc_psnr_tensor(images_target[i], sr[i])
                #     ssim_target[i] = calc_ssim_tensor(images_target[i], sr[i])
                # ssim_loss = F.mse_loss(ssim_pred, ssim_target)
                # loss = self.loss(sr, images_target) + ssim_loss
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
                timer_model.hold()
                timer_data.tic()
            # self.optimizer.schedule()
            self.running_loss_train /= (len(self.loader_train) - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.PATH_CKPOINT)
            print('MODEL SAVED at epoch: ' + str(i + 1))
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
            for batch, (data_i, image_sequence, image_label) in enumerate(self.loader_test):
                # images = [[pair],[pair],...]
                # roll a die to pick an int from range(0, self.args.num_queries)
                # idx_scale = random.randint(0, self.args.num_queries - 1)
                timer_data, timer_model = utility.timer(), utility.timer()
                image_label = construct_tensor_num_to_class(image_sequence, num_class=2)
                images_input = data_i.to(self.device)  # [pair]
                image_label = image_label.to(self.device)  # [pair]
                timer_data.hold()
                timer_model.tic()
                # sr, decoder_output = self.model(x = images[0],params=params,labels=labels)
                # sr = utility.quantize(sr, self.args.rgb_range)

                # loss = self.loss(sr, images[1]) + contrastive_loss(decoder_output, weights=0.1)
                image_label_pred = self.model(image=images_input)
                loss = self.loss(image_label_pred, image_label)
                if torch.isnan(loss):
                    num_nan += 1
                else:
                    self.running_loss_test += loss.item()
                timer_model.hold()
                timer_data.tic()

                # evaluation metrics
                tg = image_label.detach()
                pred = image_label_pred.detach()

                _, predicted = torch.max(pred.data, 1)
                _, label_target = torch.max(tg.data, 1)

                # count number of positive cases
                for i in range(label_target.size(0)):
                    if label_target[i] == 1:
                        num_positive += 1
                        if predicted[i] == 1:
                            correct_positive += 1
                        elif predicted[i] == 0:
                            false_negative += 1
                    elif label_target[i] == 0:
                        num_negative += 1
                        if predicted[i] == 1:
                            false_positive += 1
                        elif predicted[i] == 0:
                            correct_negative += 1

                total += label_target.size(0)
                correct += (predicted == label_target).sum().item()

            #     # print('tg.shape:', tg.shape)
            #     i_nmse = 0
            #     i_psnr = 0
            #     i_ssim = 0
            #     for i in range(tg.shape[0]):
            #         i_nmse = calc_nmse_tensor(tg[i], pred[i])
            #         i_psnr = calc_psnr_tensor(tg[i], pred[i])
            #         i_ssim = calc_ssim_tensor(tg[i], pred[i])
            #
            #     nmse += i_nmse/tg.shape[0]
            #     psnr += i_psnr/tg.shape[0]
            #     ssim += i_ssim/tg.shape[0]
            #
            # nmse /= len(self.loader_test)
            # psnr /= len(self.loader_test)
            # ssim /= len(self.loader_test)

            self.running_loss_test /= (len(self.loader_test) - num_nan)
            accuracy = 100. * correct / total
            positive_accuracy = 100. * correct_positive / num_positive
            negative_accuracy = 100. * correct_negative / num_negative

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| correct: ' + str(correct) + '|| total: ' + str(
                  total) + '|| accuracy: ' + str(accuracy))
        print('----------------------------------------------------------------------')
        print(
            '### detection details: num of positive: ' + str(num_positive) + '|| num of negative: ' + str(num_negative)
            + '|| correct_positive : ' + str(correct_positive) + '|| false_positive : ' + str(false_positive)
            + '|| correct_negative : ' + str(correct_negative) + '|| false_negative: ' + str(
                false_negative) + '|| positive_accuracy: ' + str(positive_accuracy) +
            '|| negative_accuracy: ' + str(negative_accuracy))
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

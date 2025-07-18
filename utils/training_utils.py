import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.help_func import print_var_detail
from utils.data_utils import img_augment
from nets.mae_gan.adap_weight import aw_loss
import time


def pretrain_mae(
        train_dataloader,
        optimizer,
        net,
        device,
        mask_ratio,
        PATH_MODEL,
        NUM_EPOCH=50,
        RESUME_EPOCH=0,
):
    '''
    Pretrain MAe model.
    :param dataloader_train:
    :param dataloader_test:
    :param optimizer:
    :param net:
    :param device:
    :param mask_ratio:
    :param NUM_EPOCH:
    :return:
    '''

    net = net.to(device)
    if RESUME_EPOCH > 0:
        net.load_state_dict(torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
        optimizer.load_state_dict(
            torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
    net.train()

    pbar = tqdm(range(RESUME_EPOCH, NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0

        start = time.time()
        for idx, data in enumerate(train_dataloader):
            if len(data) == 2:
                img, _ = data
            else:
                if isinstance(data, tuple):
                    img = data[0]
                else:
                    img = data
            img = img.to(device).float()
            if i == 0 and idx == 0:
                print('img.shape:', img.shape)
                print_var_detail(img)

            optimizer.zero_grad()
            loss, _, _ = net(img, mask_ratio)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if idx % 1000 == 0 or idx < 20:
                end = time.time()
                print('trained batches:', idx)
                print('time: ' + str(end - start) + ' sec')
                start = time.time()

        running_loss /= len(train_dataloader)
        pbar.set_description('EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, NUM_EPOCH, running_loss))
        if i == 0 or (i + 1) % (NUM_EPOCH // 5) == 0:
            print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, NUM_EPOCH, running_loss))
            # save model ckpt
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
            print('MODEL CKPT SAVED.')

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL + 'model.pt')
    print('MODEL SAVED.')

    return net


def pretrain_mae_gan(
        train_dataloader,
        optimizer,
        net,
        device,
        mask_ratio,
        PATH_MODEL,
        NUM_EPOCH=50,
        training_mode='combined',
):
    '''
    Pretrain MAe-GAN model.
    :param dataloader_train:
    :param dataloader_test:
    :param optimizer:
    :param net:
    :param device:
    :param mask_ratio:
    :param NUM_EPOCH:
    :param training_mode: 'two-stage', 'adversarial' or 'combined'
    :return:
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss_mae = 0.0
        running_loss_adv = 0.0
        running_loss_gen = 0.0
        running_loss_disc = 0.0
        running_loss_combined = 0.0
        for idx, data in enumerate(train_dataloader):
            img, _ = data
            img = img.to(device).float()
            if i == 0 and idx == 0:
                print('img.shape:', img.shape)
                print_var_detail(img)

            optimizer.zero_grad()

            if training_mode == 'two-stage':
                # backward generator loss (mae)
                mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = net(img, mask_ratio)
                mae_loss.backward()
                optimizer.step()
                running_loss_mae += mae_loss.item()
                optimizer.zero_grad()
                # backward discriminator loss
                mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = net(img, mask_ratio)
                disc_loss.backward()
                optimizer.step()  # TO: might need to see if this should be a separate optimizer
                running_loss_disc += disc_loss.item()
            elif training_mode == 'adversarial':
                # backward generator loss (adv)
                mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = net(img, mask_ratio)
                gen_loss = aw_loss(mae_loss, adv_loss, optimizer, net)
                gen_loss.backward()
                optimizer.step()
                running_loss_gen += gen_loss.item()
                optimizer.zero_grad()
                # backward discriminator loss
                mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = net(img, mask_ratio)
                disc_loss.backward()
                optimizer.step()  # TO: might need to see if this should be a separate optimizer
                running_loss_disc += disc_loss.item()
            else:
                # backward combined loss
                mae_loss, pred, mask, disc_loss, adv_loss, currupt_img = net(img, mask_ratio)
                gen_loss = aw_loss(mae_loss, adv_loss, optimizer, net)
                combined_loss = gen_loss + 2 * disc_loss
                combined_loss.backward()
                optimizer.step()
                running_loss_gen += gen_loss.item()
                running_loss_disc += disc_loss.item()
                running_loss_combined += combined_loss.item()

        running_loss_mae /= len(train_dataloader)
        running_loss_adv /= len(train_dataloader)
        running_loss_gen /= len(train_dataloader)
        running_loss_disc /= len(train_dataloader)
        running_loss_combined /= len(train_dataloader)
        if training_mode == 'two-stage':
            pbar.set_description('EPOCH [%d / %d] || AVG MAE LOSS: %f || AVG DISC LOSS: %f'
                                 % (i + 1, NUM_EPOCH, running_loss_mae, running_loss_disc))
            print('EPOCH [%d / %d] || AVG MAE LOSS: %f || AVG DISC LOSS: %f'
                  % (i + 1, NUM_EPOCH, running_loss_mae, running_loss_disc))
        elif training_mode == 'adversarial':
            pbar.set_description('EPOCH [%d / %d] || AVG GEN LOSS: %f || AVG DISC LOSS: %f'
                                 % (i + 1, NUM_EPOCH, running_loss_gen, running_loss_disc))
            print('EPOCH [%d / %d] || AVG GEN LOSS: %f || AVG DISC LOSS: %f'
                  % (i + 1, NUM_EPOCH, running_loss_gen, running_loss_disc))
        else:
            pbar.set_description('EPOCH [%d / %d] || AVG COMB LOSS: %f || AVG GEN LOSS: %f || AVG DISC LOSS: %f'
                                 % (i + 1, NUM_EPOCH, running_loss_combined, running_loss_gen, running_loss_disc))
            print('EPOCH [%d / %d] || AVG COMB LOSS: %f || AVG GEN LOSS: %f || AVG DISC LOSS: %f'
                  % (i + 1, NUM_EPOCH, running_loss_combined, running_loss_gen, running_loss_disc))
        # pbar.set_description('EPOCH [%d / %d] || AVG MAE LOSS: %f || AVG GEN LOSS: %f || AVG DISC LOSS: %f' % (i + 1, NUM_EPOCH, running_loss_mae, running_loss_gen, running_loss_disc))
        if i == 0 or (i + 1) % (NUM_EPOCH // 20) == 0:
            # print('*** EPOCH [%d / %d] || AVG MAE LOSS: %f || AVG GEN LOSS: %f || AVG DISC LOSS: %f' % (i + 1, NUM_EPOCH, running_loss_mae, running_loss_gen, running_loss_disc))
            if training_mode == 'two-stage':
                print('EPOCH [%d / %d] || AVG MAE LOSS: %f || AVG DISC LOSS: %f'
                      % (i + 1, NUM_EPOCH, running_loss_mae, running_loss_disc))
            elif training_mode == 'adversarial':
                print('EPOCH [%d / %d] || AVG GEN LOSS: %f || AVG DISC LOSS: %f'
                      % (i + 1, NUM_EPOCH, running_loss_gen, running_loss_disc))
            else:
                print('EPOCH [%d / %d] || AVG COMB LOSS: %f || AVG GEN LOSS: %f || AVG DISC LOSS: %f'
                      % (i + 1, NUM_EPOCH, running_loss_combined, running_loss_gen, running_loss_disc))
            # save model ckpt
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
            print('MODEL CKPT SAVED.')

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL + 'model.pt')
    print('MODEL SAVED.')

    return net


def pretrain_mae_per_sample_loss(
        train_dataloader,
        optimizer,
        net,
        device,
        mask_ratio,
        PATH_MODEL,
        NUM_EPOCH=50,
        RESUME_EPOCH=0,
):
    '''
    Pretrain MAe model.
    :param dataloader_train:
    :param dataloader_test:
    :param optimizer:
    :param net:
    :param device:
    :param mask_ratio:
    :param NUM_EPOCH:
    :return:
    '''

    net = net.to(device)
    if RESUME_EPOCH > 0:
        net.load_state_dict(torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
        optimizer.load_state_dict(
            torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
    net.train()

    pbar = tqdm(range(RESUME_EPOCH, NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        train_dataloader.sampler.set_epoch(i)  # Shuffle per epoch
        start = time.time()
        for idx, (img, sample_weights) in enumerate(train_dataloader):
            img = img.to(device).float()
            sample_weights = sample_weights.to(device).float()  # Shape: [B]
            if i == 0 and idx == 0:
                print('img.shape:', img.shape)
                print_var_detail(img)

            optimizer.zero_grad()
            loss_per_sample, _, _ = net(img, mask_ratio)  # loss shape: [B]

            # Apply per-sample weights
            weighted_loss = (loss_per_sample * sample_weights).mean()

            weighted_loss.backward()
            optimizer.step()
            running_loss += weighted_loss.item()

            if idx % 1000 == 0 or idx < 20:
                end = time.time()
                print('trained batches:', idx)
                print('time: ' + str(end - start) + ' sec')
                start = time.time()

        running_loss /= len(train_dataloader)
        pbar.set_description('EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, NUM_EPOCH, running_loss))
        if i == 0 or (i + 1) % (NUM_EPOCH // NUM_EPOCH) == 0:  # save every epoch
            print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, NUM_EPOCH, running_loss))
            # save model ckpt
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
            print('MODEL CKPT SAVED.')

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL + 'model.pt')
    print('MODEL SAVED.')

    return net


# def pretrain_mae_per_sample_loss(
#         train_dataloader,
#         optimizer,
#         net,
#         device,
#         mask_ratio,
#         PATH_MODEL,
#         NUM_EPOCH=50,
#         RESUME_EPOCH=0,
#         is_distributed=False,
# ):
#     '''
#     Pretrain MAe model.
#     :param dataloader_train:
#     :param dataloader_test:
#     :param optimizer:
#     :param net:
#     :param device:
#     :param mask_ratio:
#     :param NUM_EPOCH:
#     :return:
#     '''
#     net = net.to(device)
#     if RESUME_EPOCH > 0:
#         net.load_state_dict(torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
#         optimizer.load_state_dict(
#             torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
#     net.train()
#
#     pbar = tqdm(range(RESUME_EPOCH, NUM_EPOCH), desc='LOSS')
#     for i in pbar:
#         running_loss = 0.0
#         if is_distributed:
#             train_dataloader.sampler.set_epoch(i)  # Shuffle per epoch
#         start = time.time()
#         for idx, (img, sample_weights) in enumerate(train_dataloader):
#             img = img.to(device).float()
#             sample_weights = sample_weights.to(device).float()  # Shape: [B]
#             if i == 0 and idx == 0:
#                 print('img.shape:', img.shape)
#                 print_var_detail(img)
#
#             optimizer.zero_grad()
#             loss_per_sample, _, _ = net(img, mask_ratio)  # loss shape: [B]
#
#             # Apply per-sample weights
#             weighted_loss = (loss_per_sample * sample_weights).mean()
#
#             weighted_loss.backward()
#             optimizer.step()
#             running_loss += weighted_loss.item()
#
#             if idx % 1000 == 0 or idx < 20:
#                 end = time.time()
#                 print('trained batches:', idx)
#                 print('time: ' + str(end - start) + ' sec')
#                 start = time.time()
#
#         running_loss /= len(train_dataloader)
#         pbar.set_description('EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, NUM_EPOCH, running_loss))
#         if i == 0 or (i + 1) % (NUM_EPOCH // NUM_EPOCH) == 0:  # save every epoch
#             print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, NUM_EPOCH, running_loss))
#             # save model ckpt
#             torch.save({
#                 'model_state_dict': net.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }, PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
#             print('MODEL CKPT SAVED.')
#
#     # save model
#     torch.save({
#         'model_state_dict': net.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }, PATH_MODEL + 'model.pt')
#     print('MODEL SAVED.')
#
#     return net


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import time
import os


def pretrain_mae_per_sample_loss_multiGPU(
        train_dataloader,
        optimizer,
        net,
        device,
        mask_ratio,
        PATH_MODEL,
        NUM_EPOCH=50,
        RESUME_EPOCH=0,
        is_distributed=False,
):
    rank = dist.get_rank() if is_distributed else 0
    is_main_process = (rank == 0)

    # # Move model to GPU and wrap in DDP if distributed
    # net = net.to(device)
    # if is_distributed:
    #     net = DDP(net, device_ids=[device], output_device=device)

    # if RESUME_EPOCH > 0:
    #     ckpt = torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt', map_location=device)
    #     net.load_state_dict(ckpt['model_state_dict'])
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    net = net.to(device)

    # Load checkpoint before wrapping with DDP
    if RESUME_EPOCH > 0:
        ckpt_path = os.path.join(PATH_MODEL, f'model_E{RESUME_EPOCH}.pt')
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # Now wrap with DDP
    if is_distributed:
        net = DDP(net, device_ids=[device.index], output_device=device.index)

    net.train()

    if is_main_process:
        pbar = tqdm(range(RESUME_EPOCH, NUM_EPOCH), desc='LOSS')
    else:
        pbar = range(RESUME_EPOCH, NUM_EPOCH)

    for i in pbar:
        running_loss = 0.0
        if is_distributed:
            train_dataloader.sampler.set_epoch(i)

        start = time.time()

        for idx, (img, sample_weights) in enumerate(train_dataloader):
            img = img.to(device).float()
            sample_weights = sample_weights.to(device).float()

            if is_main_process and i == 0 and idx == 0:
                print('img.shape:', img.shape)

            optimizer.zero_grad()
            loss_per_sample, _, _ = net(img, mask_ratio)
            weighted_loss = (loss_per_sample * sample_weights).mean()
            weighted_loss.backward()
            optimizer.step()

            running_loss += weighted_loss.item()

            if is_main_process and (idx % 1000 == 0 or idx < 20):
                print(f'[Epoch {i}] Batch {idx} — Time: {time.time() - start:.2f}s')
                start = time.time()

        running_loss /= len(train_dataloader)

        if is_main_process:
            pbar.set_description(f'EPOCH [{i + 1} / {NUM_EPOCH}] || AVG LOSS: {running_loss:.6f}')

            # Save checkpoint every epoch
            torch.save({
                'model_state_dict': net.module.state_dict() if is_distributed else net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(PATH_MODEL, f'model_E{i + 1}.pt'))
            print(f'[Epoch {i + 1}] Checkpoint saved.')
            # Log loss to file
            log_path = os.path.join(PATH_MODEL, "loss_log.txt")
            with open(log_path, "a") as f:
                f.write(f"Epoch {i + 1}: loss = {running_loss:.6f}\n")

    # Save final model
    if is_main_process:
        torch.save({
            'model_state_dict': net.module.state_dict() if is_distributed else net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(PATH_MODEL, 'model.pt'))
        print('Final model saved.')

    return net
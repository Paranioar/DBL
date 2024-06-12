"""
# Pytorch implementation for TIP2024 paper from
# https://arxiv.org/abs/2404.18114.
# "Deep Boosting Learning: A Brand-new Cooperative Approach for Image-Text Matching"
# Haiwen Diao, Ying Zhang, Shang Gao, Xiang Ruan, Huchuan Lu
#
# Writen by Haiwen Diao, 2021
"""

import os
import time
import shutil

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from evaluation import evalrank, AverageMeter, LogCollector
from loss import init_loss

import logging
import collections
import tensorboard_logger as tb_logger


def main():
    # Hyper Parameters
    opt = opts.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid
    device_count = len(str(opt.gpuid).split(","))

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger_name = os.path.join(opt.save_path, 'log')
    tb_logger.configure(logger_name, flush_secs=5)
    model_name = os.path.join(opt.save_path, 'checkpoint')
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    model_list = []
    optimizer_list = []
    # Construct the multi-model
    for i in range(opt.num_branch):
        model = SGRAF(opt)
        model.cuda()
        model = nn.DataParallel(model)
        model.Eiters = 0

        model_list.append(model)
        optimizer_list.append(torch.optim.Adam(model.parameters(), lr=opt.learning_rate))

    # Train the Model
    best_rsum = [0] * len(model_list)
    criterion = init_loss(name='birank', margin=opt.margin, max_violation=opt.max_violation)
    criterion_boo = init_loss(name=opt.boost_name, margin=opt.margin, beta=opt.beta)
    criterion_list = [criterion, criterion_boo]

    lr_schedules = [opt.lr_update, ]

    # Start the training
    for epoch in range(opt.num_epochs):
        print(logger_name)
        print(model_name)

        adjust_learning_rate(optimizer_list, epoch, lr_schedules)
        train_logger_list = []
        for k in range(opt.num_branch):
            train_logger_list.append(LogCollector())

        for i, (images, captions, lengths, ids, img_ids) in enumerate(train_loader):

            if device_count != 1:
                images = images.repeat(device_count, 1, 1)
            sims_list = []
            for j, model in enumerate(model_list):
                # Switch to train mode
                model.train()
                sims = model(images, captions, lengths)
                sims_list.append(sims.permute(1, 0))

            for m, model in enumerate(model_list):
                # Log the training process
                model.logger = train_logger_list[m]
                model.Eiters += 1
                model.logger.update('Eit_branch(%d)' % m, model.Eiters)
                model.logger.update('Lr_branch(%d)' % m, optimizer_list[m].param_groups[0]['lr'])

                # Update the model
                optimizer_list[m].zero_grad()
                loss = compute_loss(opt, m, sims_list, criterion_list, img_ids, model.logger)
                loss.backward()
                if opt.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), opt.grad_clip)
                optimizer_list[m].step()

                # Print log info
                if model.Eiters % opt.log_step == 0:
                    logging.info(
                        'Epoch: [{0}][{1}/{2}]\t'
                        '{e_log}\t'
                        .format(epoch, i, len(train_loader), e_log=str(model.logger)))

                # Record logs in tensorboard
                tb_logger.log_value('epoch', epoch, step=model.Eiters)
                tb_logger.log_value('step', i, step=model.Eiters)
                model.logger.tb_log(tb_logger, step=model.Eiters)

        for n, model in enumerate(model_list):
            # evaluate on validation set
            print("-------------------------------------")
            print('Validate the {0} branch at {1} epoch:'.format(n, epoch))
            rsum = evalrank(model.module, n, val_loader, opt, step=model.Eiters)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum[n]
            best_rsum[n] = max(rsum, best_rsum[n])
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum[n],
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, n, filename='checkpoint{}_branch{}.pth.tar'.format(epoch, n), prefix=model_name + '/')


def compute_loss(opt, model_index, sims_list, criterion_list, img_ids, logger):

    loss_raw = criterion_list[0](sims_list[model_index])
    logger.update('Loss_raw_branch(%d)' % model_index, loss_raw.item(), sims_list[0].size(0))

    loss_boo = 0
    if opt.num_branch > 1 and model_index:
        for n in range(model_index):
            loss_boo += criterion_list[1](sims_list[model_index], sims_list[n].detach())
        loss_boo = loss_boo / model_index
        logger.update('Loss_boo_branch(%d)' % model_index, loss_boo.item(), sims_list[0].size(0))

    loss = loss_raw + opt.alpha * loss_boo
    logger.update('Loss_branch(%d)' % model_index, loss.item(), sims_list[0].size(0))
    return loss


def save_checkpoint(state, is_best, model_index, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model{}_best.pth.tar'.format(model_index))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(optimizer_list, epoch, lr_schedules):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    if epoch in lr_schedules:
        for optimizer in optimizer_list:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * 0.1
                param_group['lr'] = new_lr


if __name__ == '__main__':
    main()
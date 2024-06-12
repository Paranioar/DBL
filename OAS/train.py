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
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from evaluation import evalrank, AverageMeter, LogCollector
from loss import init_loss
from utils import F30kSpacySimMat

import logging
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
    vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SGRAF(opt)
    model.cuda()
    model = nn.DataParallel(model)

    # Train the Model
    model.Eiters = 0
    best_rsum = 0

    criterion = init_loss(name='birank', margin=opt.margin, max_violation=opt.max_violation)
    criterion_boo = init_loss(name=opt.boost_name, margin=opt.margin, beta=opt.beta)
    criterion_list = [criterion, criterion_boo]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, checkpoint['epoch'], checkpoint['best_rsum']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Construct the teacher model
    if opt.if_boost and opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
    
            # Construct the teacher model
            model_t = SGRAF(opt)
            model_t.cuda()
            model_t = nn.DataParallel(model_t)
            model_t.load_state_dict(checkpoint['model'])
            for p in model_t.parameters():
                p.requires_grad = False
            print('Now initializing the teacher model...')
    
            # Eiters is used to show logs as the continuation of another training
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, checkpoint['epoch'], checkpoint['best_rsum']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Start the training
    for epoch in range(opt.num_epochs):
        print(logger_name)
        print(model_name)

        adjust_learning_rate(opt, optimizer, epoch)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()

        end = time.time()
        for i, (images, captions, lengths, ids, img_ids) in enumerate(train_loader):
            # switch to train mode
            model.train()

            # measure data loading time
            data_time.update(time.time() - end)
            model.logger = train_logger

            model.Eiters += 1
            model.logger.update('Eit', model.Eiters)
            model.logger.update('Lr', optimizer.param_groups[0]['lr'])
            # Update the model
            optimizer.zero_grad()

            if device_count != 1:
                images = images.repeat(device_count, 1, 1)
            sims_list = []
            sims = model(images, captions, lengths)
            sims_list.append(sims.permute(1, 0))

            # similarity computed by teacher model
            if opt.if_boost and opt.resume:
                with torch.no_grad():
                    model_t.eval()
                    sims_t = model_t(images, captions, lengths)
                    sims_list.append(sims_t.permute(1, 0))

            loss = compute_loss(opt, sims_list, criterion_list, ids, img_ids, model.logger)
            loss.backward()

            if opt.grad_clip > 0:
                clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print log info
            if model.Eiters % opt.log_step == 0:
                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(epoch, i, len(train_loader), batch_time=batch_time,
                                data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)

        # evaluate on validation set
        rsum = evalrank(model.module, val_loader, opt, step=model.Eiters)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=model_name + '/')


def compute_loss(opt, sims_list, criterion_list, ids, img_ids, logger):
    loss = criterion_list[0](sims_list[0])

    if len(sims_list) == 1:
        logger.update('Loss', loss.item(), sims_list[0].size(0))
        return loss
    else:
        loss_boo = criterion_list[1](sims_list[0], sims_list[1])

        logger.update('Loss_raw', loss.item(), sims_list[0].size(0))
        logger.update('Loss_boo', loss_boo.item(), sims_list[0].size(0))

        loss += opt.alpha * loss_boo
        logger.update('Loss', loss.item(), sims_list[0].size(0))
        return loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

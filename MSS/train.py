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

    # Construct the model
    model = SGRAF(opt)
    model.cuda()
    model = nn.DataParallel(model)

    # Construct the teacher model
    model_t = SGRAF(opt)
    model_t.cuda()
    model_t = nn.DataParallel(model_t)
    model_t.load_state_dict(model.state_dict())
    print('Now initializing the teacher model...')
    for p in model_t.parameters():
        p.requires_grad = False

    # Train the Model
    model.Eiters = 0
    best_rsum = 0

    criterion = init_loss(name='birank', margin=opt.margin, max_violation=opt.max_violation)
    criterion_boo = init_loss(name=opt.boost_name, margin=opt.margin, beta=opt.beta)
    criterion_list = [criterion, criterion_boo]

    # optionally resume from a checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    momentum_schedule = cosine_scheduler(opt.momentum_teacher, 1,
                                         opt.num_epochs, len(train_loader))

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
            it = len(train_loader) * epoch + i

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
            with torch.no_grad():
                model_t.eval()
                sims_t = model_t(images, captions, lengths)
                sims_list.append(sims_t.permute(1, 0))

            loss = compute_loss(opt, sims_list, criterion_list, img_ids, model.logger)
            loss.backward()

            if opt.grad_clip > 0:
                clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(model.module.parameters(), model_t.module.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # measure elapsed time
            torch.cuda.synchronize()
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
        print("-----------------------------------")
        print('Validate the student model at {} epoch:'.format(epoch))
        rsum = evalrank(model.module, val_loader, opt, step=model.Eiters)
        print("-----------------------------------")
        print('Validate the teacher model at {} epoch:'.format(epoch))
        evalrank(model_t.module, val_loader, opt, step=model.Eiters)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'model_t': model_t.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=model_name + '/')


def compute_loss(opt, sims_list, criterion_list, img_ids, logger):
    loss_stu = criterion_list[0](sims_list[0])
    loss_tea = criterion_list[0](sims_list[1])
    loss_boo = criterion_list[1](sims_list[0], sims_list[1])
    loss = loss_stu + opt.alpha * loss_boo

    logger.update('Loss_stu', loss_stu.item(), sims_list[0].size(0))
    logger.update('Loss_tea', loss_tea.item(), sims_list[0].size(0))
    logger.update('Loss_boo', loss_boo.item(), sims_list[0].size(0))
    logger.update('Loss', loss.item(), sims_list[0].size(0))
    return loss


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


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

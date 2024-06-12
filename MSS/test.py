import evaluation as evaluation
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
import data

import argparse
import os
import torch
import torch.nn as nn


def test(model_path, split, gpuid='0', fold5=False):
    print("use GPU:", gpuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = SGRAF(opt)
    model.cuda()
    model = nn.DataParallel(model)

    # load model state
    model.load_state_dict(checkpoint['model'])
    # model.load_state_dict(checkpoint['model_t'])
    data_loader = data.get_test_loader(split, opt.data_name, vocab,
                                       opt.batch_size, opt.workers, opt)

    print(opt)
    print('Computing results with checkpoint_{}'.format(checkpoint['epoch']))

    evaluation.evalrank(model.module, data_loader, opt, split, fold5)


if __name__ == '__main__':
    # F30K
    test('runs/model_name/checkpoint/model_best.pth.tar', 'test', '0', False)
    # COCO1K
    test('runs/model_name/checkpoint/model_best.pth.tar', 'testall', '0', True)
    # COCO5K
    test('runs/model_name/checkpoint/model_best.pth.tar', 'testall', '0', False)






import argparse
import yaml
import os
from ast import literal_eval as make_tuple

import torch
import utils


def add_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # options overwritting yaml options
    parser.add_argument('--path_opt', default='default.yaml',
                        type=str, help='path to a yaml options file')
    parser.add_argument('--data', default=argparse.SUPPRESS,
                        type=str, help='path to data')
    parser.add_argument('--logger_name', default='runs/runX')
    parser.add_argument('--dataset', default='mnist', help='mnist|cifar10')

    # options that can be changed from default
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size',
                        type=int, default=argparse.SUPPRESS, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=argparse.SUPPRESS,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=argparse.SUPPRESS,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='how many batches to wait before logging training'
                        ' status')
    parser.add_argument('--tblog_interval',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--optim', default=argparse.SUPPRESS, help='sgd|dmom')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default=argparse.SUPPRESS,
                        help='model architecture: (default: resnet32)')
    parser.add_argument('-j', '--workers', default=argparse.SUPPRESS,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--weight_decay', '--wd', default=argparse.SUPPRESS,
                        type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--train_accuracy', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--log_profiler', action='store_true')
    parser.add_argument('--lr_decay_epoch',
                        default=argparse.SUPPRESS)
    parser.add_argument('--log_keys', default='')
    parser.add_argument('--exp_lr',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--no_transform',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--corrupt_perc',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--log_nex',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--data_aug',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--wnoise',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--wnoise_stddev',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--noresume', action='store_true',
                        help='resume by default if an old run exists.')
    parser.add_argument('--g_noresume', action='store_true',
                        help='resume by default if an old run exists.')
    parser.add_argument('--ckpt_name', default='model_best.pth.tar')
    parser.add_argument('--pretrained',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--nodropout',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--nobatchnorm',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--num_class',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--lr_decay_rate',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--nesterov',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--label_smoothing',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--duplicate',
                        default=argparse.SUPPRESS, type=str)
    parser.add_argument('--g_estim', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--epoch_iters',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_log_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_estim_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_debug',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--gvar_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_bsnap_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_osnap_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_optim',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_optim_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_optim_start_plus', action='store_true')
    parser.add_argument('--half_trained', action='store_true')
    parser.add_argument('--g_epoch',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_batch_size',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--adam_betas',
                        default=argparse.SUPPRESS, type=str)
    parser.add_argument('--adam_eps',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_mlr',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--svrg_bsnap_num',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--niters',
                        default=argparse.SUPPRESS, type=int)
    # logreg
    parser.add_argument('--num_train_data',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--num_test_data',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--dim',
                        default=argparse.SUPPRESS, type=int)
    args = parser.parse_args()
    return args


def yaml_opt(yaml_path):
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle, Loader=yaml.FullLoader)
    return opt


def get_opt():
    args = add_args()
    opt = yaml_opt('options/default.yaml')
    opt_s = yaml_opt(os.path.join('options/{}/{}'.format(args.dataset,
                                                         args.path_opt)))
    opt.update(opt_s)
    opt.update(vars(args).items())
    opt = utils.DictWrapper(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    if opt.g_batch_size == -1:
        opt.g_batch_size = opt.batch_size
    opt.adam_betas = make_tuple(opt.adam_betas)
    return opt

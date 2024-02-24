import argparse
import os
import torch

import numpy as np
import torch
import random

import re
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset',
        required=False,
        type=str,
        default="esc10",
        help="Define the dataset to be used for training (default=esc10, valid=[esc10, esc50, dcase19, fsc])"
    )

    parser.add_argument(
        '-n',
        '--num_epochs',
        required=False,
        type=int,
        default=1,
        help="Define the number of epochs to train the model for (default=1, valid=int)"
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        required=False,
        type=int,
        default=2,
        help="Define the batch size for training (defaultt=2, valid=int)"
    )

    parser.add_argument(
        '-nl',
        '--new_loss',
        required=False,
        type=bool,
        default=True,
        help="Let us know if you want to use the new loss function (default=True, valid=[True, False])"
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help="Define the device to train the model in (default=cuda if available, else cpu, valid=[cpy, cuda])")

    parser.add_argument(
        '--debug',
        required=False,
        type=bool,
        default=False,
        help="Show debug data. Shows values of arguments (default=False, valid=[True, False])"
    )

    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--debug_subset_size', type=int, default=8)
    # parser.add_argument('--download', action='store_true',
    #                     help="if can't find dataset, download from web")
    # parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    # parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    # parser.add_argument('--ckpt_dir', type=str,
    #                     default=os.getenv('CHECKPOINT'))
    # parser.add_argument('--ckpt_dir_1', type=str,
    #                     default=os.getenv('CHECKPOINT'))
    # parser.add_argument('--eval_from', type=str, default=None)
    # parser.add_argument('--hide_progress', action='store_true')
    # parser.add_argument('--cl_default', action='store_true')
    # parser.add_argument('--server', action='store_true')
    # parser.add_argument('--hcl', action='store_true')
    # parser.add_argument('--buffer_qdi', action='store_true')
    # parser.add_argument('--validation', action='store_true',
    #                     help='Test on the validation set')
    # parser.add_argument('--ood_eval', action='store_true',
    #                     help='Test on the OOD set')
    # parser.add_argument('--alpha', type=float, default=0.3)
    args = parser.parse_args()
    if args.debug:
        print(f"Using dataset {args.dataset}")
        print(f"Using number of epochs {args.num_epochs}")
        print(f"Using batch size {args.batch_size}")
        print(f"{'Not ' if not args.new_loss else ''}Using New Loss Function")
        print(f"Using device {args.device}\n")

    # with open(args.config_file, 'r') as f:
    #     for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
    #         vars(args)[key] = value

    # if args.debug:
    #     if args.train:
    #         args.train.batch_size = 2
    #         args.train.num_epochs = 1
    #         args.train.stop_at_epoch = 1
    #     if args.eval:
    #         args.eval.batch_size = 2
    #         args.eval.num_epochs = 1  # train only one epoch
    #     args.dataset.num_workers = 0

    # assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    # args.log_dir = os.path.join(
    #     args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)

    # os.makedirs(args.log_dir, exist_ok=False)
    # print(f'creating file {args.log_dir}')
    # os.makedirs(args.ckpt_dir, exist_ok=True)

    # shutil.copy2(args.config_file, args.log_dir)
    # set_deterministic(args.seed)

    # vars(args)['aug_kwargs'] = {
    #     'name': args.model.name,
    #     'image_size': args.dataset.image_size,
    #     'cl_default': args.cl_default
    # }
    # vars(args)['dataset_kwargs'] = {
    #     # 'name':args.model.name,
    #     # 'image_size': args.dataset.image_size,
    #     'dataset': args.dataset.name,
    #     'data_dir': args.data_dir,
    #     'download': args.download,
    #     'debug_subset_size': args.debug_subset_size if args.debug else None,
    #     # 'drop_last': True,
    #     # 'pin_memory': True,
    #     # 'num_workers': args.dataset.num_workers,
    # }
    # vars(args)['dataloader_kwargs'] = {
    #     'drop_last': True,
    #     'pin_memory': True,
    #     'num_workers': args.dataset.num_workers,
    # }

    return args

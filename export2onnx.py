
import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset, OpenImagesDataset2, OpenImagesDataset3
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

import torchvision.transforms.functional as FT
import torchvision.transforms as T
from thop import profile, clever_format
from flopth import flopth
import time

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


if __name__ == '__main__':
    timer = Timer()

    class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
    class_dict = {"qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5, "background": 0}
    label2class_dict = {1:"qrcode", 2:"barcode", 3:"mpcode", 4:"pdf417", 5:"dmtx", 0:"background"}
    args.net = 'mb3-small-ssd-lite'
    args.net = 'mb2-ssd-lite'
    args.net = 'mb1-ssd'
    args.checkpoint_folder = os.path.join(os.getcwd(),'checkpoint')
    args.dataset_type = 'xcode'
    args.datasets = [os.path.join(os.getcwd(),'jsons')]
    # model_path = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\mb3-small-ssd-lite-Epoch-50-Loss-6.65807689583105.pth"

    pretrained_model_path = None

    # args.scheduler = None
    # args.resume = r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\mb3-small-ssd-lite-Epoch-50-Loss-6.65807689583105.pth"

    # args.net = 'mb3-small-ssd-lite'
    # args.checkpoint_folder = os.path.join(os.getcwd(),'checkpoint')
    # args.dataset_type = 'voc'
    # args.datasets = ['D:\datas\VOCdevkit\VOC2007','D:\datas\VOCdevkit\VOC2012']
    # args.validation_dataset = 'D:\datas\VOCdevkit\VOC2007-test'
    # args.t_max = 200
    # args.validation_epochs = 5
    # args.num_epochs = 200
    # args.scheduler = 'cosine'
    # args.lr = 0.01

    logging.info(args)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'mb3-large-ssd-lite':
        net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb3-small-ssd-lite':
        net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    if pretrained_model_path:
        # net.load(args.trained_model)
        net.load(pretrained_model_path)
    
    # net.load_state_dict(torch.load(r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\mb3-small-ssd-lite-Epoch-15-Loss-9.15076301574707.pth"))
    # net.load_state_dict(torch.load(r"C:\kwoncy\projects\xcode-detection\pytorch-ssd\checkpoint\mb2-ssd-lite-Epoch-65-Loss-3.078031623363495.pth"))
    net.to(DEVICE)
    
    from pytorch_model_summary import summary
    dummy_input= torch.randn(1,3,300,300).to(DEVICE)
    print(summary(net, dummy_input, show_input=True))
    macs, params = profile(net, inputs=(dummy_input,))
    macs, params = clever_format([macs,params], "%.3f")
    print('thop macs: ',macs)
    print('thop params: ', params)
    flops, params = flopth(net, inputs=(dummy_input,))
    print('flopth flops: ', flops)
    print('flopth params: ', params)

    start_time = time.time()
    _ = net(dummy_input)
    print('time: ', time.time()-start_time)

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    
    

    





    
    
    

# System / Python
import os
import argparse
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# Custom
from net import ParallelKINetworkV2 as Network
from mri_tools import *
from paired_dataset import *
from utils import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default='tcp://localhost:1836', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(), help='number of gpus per node')
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True,
                    help='whether initialize model weights with defined types')
parser.add_argument('--net_G', type=str, default='DRDN', help='generator network')  # DRDN / SCNN
parser.add_argument('--n_recurrent', type=int, default=2, help='Number of reccurent block in model')
parser.add_argument('--use_prior', default=False, action='store_true', help='use prior')  # True / False
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=2, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train', metavar='/path/to/training_data',
                    default="./fastMRI_brain_DICOM/t1_t2_paired_6875_train.csv", type=str)
parser.add_argument('--val', metavar='/path/to/validation_data',
                    default="./fastMRI_brain_DICOM/t1_t2_paired_6875_val.csv", type=str)
parser.add_argument('--test', metavar='/path/to/test_data', default="./fastMRI_brain_DICOM/t1_t2_paired_6875_test.csv",
                    type=str)
parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask_8.00x_acs24.mat',
                    help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat',
                    help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat',
                    help='selection mask in down network')
parser.add_argument('--train-obj-limit', type=int, default=20, help='number of objects in training set')
parser.add_argument('--val-obj-limit', type=int, default=5, help='number of objects in val set')
parser.add_argument('--test-obj-limit', type=int, default=20, help='number of objects in test set')
parser.add_argument('--prefetch', action='store_false')
# save path
parser.add_argument('--output-path', type=str, default='./run/', help='output path')
# others
parser.add_argument('--mode', '-m', type=str, default='train',
                    help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')


def solvers(rank, ngpus_per_node, args):
    # logger and devices
    if rank == 0:
        logger = create_logger(args)
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size,
                                         rank=rank)

    # model
    model = Network(rank, args)

    # load checkpoint
    try:
        model.load()
    except:
        pass
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    if rank == 0:
        logger.info('Load checkpoint at epoch {}.'.format(model.module.epoch))
        logger.info('Current learning rate is {}.'.format(model.module.optimizer.param_groups[0]['lr']))
        logger.info('Current best metric in train phase is {}.'.format(model.module.best_target_metric))

    if args.mode == 'test':
        # data
        test_set = get_dataset_split(args, 'test')
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        if rank == 0:
            logger.info('The size of test dataset is {}.'.format(len(test_set)))

        # run one epoch
        test_log = model.module.test_one_epoch(test_loader)

        # log
        if rank == 0:
            logger.info(f'time:{test_log["time"]:.5f}s\t'
                        f'test_loss:{test_log["loss"]:.7f}\t'
                        f'test_psnr1:{test_log["psnr1"]:.7f}\t'
                        f'test_psnr2:{test_log["psnr2"]:.5f}\t'
                        f'test_ssim1:{test_log["ssim1"]:.7f}\t'
                        f'test_ssim2:{test_log["ssim2"]:.5f}\t')

    # data
    train_set, val_set = get_dataset_split(args, 'train'), get_dataset_split(args, 'val')
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    if rank == 0:
        logger.info('The size of train dataset is {}.'.format(len(train_set)))
        logger.info('The size of val dataset is {}.'.format(len(val_set)))



    # training loop
    for epoch in range(model.module.epoch + 1, args.num_epochs + 1):
        # data and run one epoch
        train_sampler.set_epoch(epoch)
        train_loader = tqdm(train_loader, desc='training', total=int(len(train_loader))) if rank == 0 else train_loader
        train_log = model.module.train_one_epoch(train_loader)

        val_loader = tqdm(val_loader, desc='valing', total=int(len(val_loader))) if rank == 0 else val_loader
        val_log = model.module.eval_one_epoch(val_loader)

        # log
        if rank == 0:
            # output log
            logger.info(f'epoch:{train_log["epoch"]:<8d}\t'
                        f'time:{train_log["time"]:.5f}s\t'
                        f'lr:{train_log["lr"]:.5f}s\t'
                        f'train_loss:{train_log["loss"]:.8f}\t'
                        f'val_loss:{val_log["loss"]:.7f}\t'
                        f'val_psnr1:{val_log["psnr1"]:.7f}\t'
                        f'val_psnr2:{val_log["psnr2"]:.5f}\t'
                        f'val_ssim1:{val_log["ssim1"]:.7f}\t'
                        f'val_ssim2:{val_log["ssim2"]:.5f}\t')

        if model.module.signal_to_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
            break


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus

    os.makedirs(args.output_path, exist_ok=True)
    args.model_save_path = os.path.join(args.output_path, 'checkpoints')
    args.loss_curve_path = os.path.join(args.output_path, 'loss_curve')
    args.log_path = os.path.join(args.output_path, 'log.txt')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))


if __name__ == '__main__':
    main()

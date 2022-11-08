# System / Python
import argparse
import logging
import random
# tool
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# Custom
from net import SemisupervisedParallelKINetworkV2 as Network
from mri_tools import *
from paired_dataset import *
from utils import *
from sampler import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
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
parser.add_argument('--output-path', type=str, default='./runs/test_run/', help='output path')
# others
parser.add_argument('--mode', '-m', type=str, default='train',
                    help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')
parser.add_argument('--resume', action='store_true', help='whether resume to train')
# semi-supervised training
parser.add_argument('--supervised-every', type=int, default=4,
                    help='One supervised subject every how many subjects, to build a semi-supervised dataset')
parser.add_argument('--supervised-mode', metavar='type', choices=['semi-supervised', 'supervised', 'self-supervised'],
                    required=True, type=str, help='types of learning')
parser.add_argument('--T-s', type=int, default=1)
parser.add_argument('--T-us', type=int, default=1)


def solvers(args):
    # logger and devices
    logger = create_logger(args)

    logger.info(args)

    # model
    model = Network(0, args)  # 0 is rank
    if args.mode == 'test':
        model.load_best()
    elif args.resume:
        model.load()
    model = model.cuda()

    logger.info('Current epoch {}.'.format(model.epoch))
    logger.info('Current learning rate is {}.'.format(model.optimizer.param_groups[0]['lr']))
    logger.info('Current best metric in train phase is {}.'.format(model.best_target_metric))

    if args.mode == 'test':
        # data
        test_set = get_dataset_split(args, 'test')
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        logger.info('The size of test dataset is {}.'.format(len(test_set)))
        test_loader = tqdm(test_loader, desc='testing', total=int(len(test_loader)))

        # run one epoch
        test_log = model.test_one_epoch(test_loader)

        # log
        logger.info(f'time:{test_log["time"]:.5f}s\t'
                    f'test_loss:{test_log["loss"]:.7f}\t'
                    f'test_psnr1:{test_log["psnr1"]:.7f}\t'
                    f'test_psnr2:{test_log["psnr2"]:.5f}\t'
                    f'test_ssim1:{test_log["ssim1"]:.7f}\t'
                    f'test_ssim2:{test_log["ssim2"]:.5f}\t')

        return

    # data
    semi_train_set, unsup_train_set, sup_train_set = get_semisupervised_dataset_split(args, 'train')

    if args.supervised_mode == 'semi-supervised':  # jointly use unsup and sup data
        assert unsup_train_set is not None and sup_train_set is not None and \
               len(unsup_train_set) > 0 and len(sup_train_set) > 0
        train_set = semi_train_set
        alt_sampler = AlternatingSampler(train_set, T_s=args.T_s, T_us=args.T_us)
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, sampler=alt_sampler, pin_memory=True)
        logger.info('Semi-supervised learning.')
        logger.info(f'Unsupervised:supervised: '
                    f'obj level: {len(train_set.unsupervised_volume_idx)}:{len(train_set.supervised_volume_idx)}, '
                    f'slice level: {len(train_set.unsupervised_idx)}:{len(train_set.supervised_idx)}.')
    elif args.supervised_mode == 'supervised':  # only use sup data
        assert sup_train_set is not None and len(sup_train_set) > 0
        train_set = sup_train_set
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        logger.info('Supervised learning.')
        logger.info('The size of train dataset is {}.'.format(len(train_set)))
    elif args.supervised_mode == 'self-supervised':  # only use unsup data
        assert unsup_train_set is not None and len(unsup_train_set) > 0
        train_set = unsup_train_set
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        logger.info('Self-supervised learning.')
        logger.info('The size of train dataset is {}.'.format(len(train_set)))

    val_set = get_dataset_split(args, 'val')
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    logger.info('The size of val dataset is {}.'.format(len(val_set)))

    # training loop
    for epoch in range(model.epoch + 1, args.num_epochs + 1):
        # data and run one epoch
        train_loader = tqdm(train_loader, desc='training', total=int(len(train_loader)))
        train_log = model.train_one_epoch(train_loader)
        val_loader = tqdm(val_loader, desc='valing', total=int(len(val_loader)))
        val_log = model.eval_one_epoch(val_loader)

        # output log
        logger.info(f'epoch:{train_log["epoch"]:<8d}\t'
                    f'time:{train_log["time"]:.2f}s\t'
                    f'lr:{train_log["lr"]:.8f}\t'
                    f'train_loss:{train_log["loss"]:.7f}\t'
                    f'val_loss:{val_log["loss"]:.7f}\t'
                    f'val_psnr1:{val_log["psnr1"]:.5f}\t'
                    f'val_psnr2:{val_log["psnr2"]:.5f}\t'
                    f'val_ssim1:{val_log["ssim1"]:.5f}\t'
                    f'val_ssim2:{val_log["ssim2"]:.5f}\t')

        if model.signal_to_stop:
            logger.info('The experiment is early stop!')
            break


def main():
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    args.model_save_path = os.path.join(args.output_path, 'checkpoints')
    args.log_path = os.path.join(args.output_path, 'log.txt')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    solvers(args)


if __name__ == '__main__':
    main()

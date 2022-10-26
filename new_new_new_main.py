# System / Python
import os
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
# from net import ParallelNetwork as Network
# from net import ParallelKINetwork as Network
# from net import ParallelDuDoRNetwork as Network
from net import ParallelKINetworkDC as Network
from net import ParallelIINetworkDC as IINetwork
from net import ParallelKKNetworkDC as KKNetwork
from IXI_dataset import IXIData as Dataset
from mri_tools import *
from utils import psnr_slice, ssim_slice
from paired_dataset import *

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
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=5, help='number of iterations')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=2, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train-path', type=str, default='./IXI_T1/train', help='path of training data')
parser.add_argument('--val-path', type=str, default='./IXI_T1/val', help='path of validation data')
parser.add_argument('--test-path', type=str, default='./IXI_T1/test', help='path of test data')
parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask_8.00x_acs24.mat', help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.02, help='sampling rate of training data')
parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.01, help='sampling rate of validation data')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.01, help='sampling rate of test data')
# save path
parser.add_argument('--output-path', type=str, default='./run/', help='output path')
# parser.add_argument('--model-save-path', type=str, default='./run/checkpoints/', help='save path of trained model')
# parser.add_argument('--loss-curve-path', type=str, default='./run/loss_curve/', help='save path of loss curve in tensorboard')
# parser.add_argument('--log-path', type=str, default='./run/log.txt', help='save path of log')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')

parser.add_argument('--net_G', type=str, default='DRDN', help='generator network')   # DRDN / SCNN
parser.add_argument('--n_recurrent', type=int, default=2, help='Number of reccurent block in model')
parser.add_argument('--use_prior', default=False, action='store_true', help='use prior')   # True / False
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')

parser.add_argument('--train', metavar='/path/to/training_data', default="./fastMRI_brain_DICOM/t1_t2_paired_6875_train.csv", type=str)
parser.add_argument('--val', metavar='/path/to/validation_data', default="./fastMRI_brain_DICOM/t1_t2_paired_6875_val.csv", type=str)
parser.add_argument('--test', metavar='/path/to/test_data', default="./fastMRI_brain_DICOM/t1_t2_paired_6875_test.csv", type=str)
parser.add_argument('--prefetch', action='store_false')
parser.add_argument('--train-obj-limit', type=int, default=20, help='number of objects in training set')
parser.add_argument('--val-obj-limit', type=int, default=5, help='number of objects in val set')
parser.add_argument('--test-obj-limit', type=int, default=20, help='number of objects in test set')

def create_logger(args):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename=args.log_path, mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Prefetch(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = [i for i in tqdm(dataset, leave=False)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind]

def get_dataset(args):
    print('loading data...')
    volumes_train = get_paired_volume_datasets(
            args.train, crop=256, protocals=['T2'],
            object_limit=args.train_obj_limit,
            u_mask_path=args.u_mask_path,
            s_mask_up_path=args.s_mask_up_path,
            s_mask_down_path=args.s_mask_down_path)
    volumes_val = get_paired_volume_datasets(
            args.val, crop=256, protocals=['T2'],
            object_limit=args.val_obj_limit,
            u_mask_path=args.u_mask_path,
            s_mask_up_path=args.s_mask_up_path,
            s_mask_down_path=args.s_mask_down_path
    )
    volumes_test = get_paired_volume_datasets(
            args.test, crop=256, protocals=['T2'],
            object_limit=args.test_obj_limit,
            u_mask_path=args.u_mask_path,
            s_mask_up_path=args.s_mask_up_path,
            s_mask_down_path=args.s_mask_down_path
    )
    slices_train = torch.utils.data.ConcatDataset(volumes_train)
    slices_val = torch.utils.data.ConcatDataset(volumes_val)
    slices_test = torch.utils.data.ConcatDataset(volumes_test)
    if args.prefetch:
        # load all data to ram
        slices_train = Prefetch(slices_train)
        slices_val = Prefetch(slices_val)
        slices_test = Prefetch(slices_test)
    # loader_train = torch.utils.data.DataLoader(
    #         slices_train, batch_size=args.batch_size, shuffle=True,
    #         num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # loader_val = torch.utils.data.DataLoader(
    #         slices_val, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # loader_test = torch.utils.data.DataLoader(
    #         slices_test, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.num_workers, pin_memory=True, drop_last=True)

    return slices_train, slices_val, slices_test


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args):
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        label = data_batch[0].to(rank, non_blocking=True)  # full sampled image [bs, 1, x, y]
        label = torch.view_as_real(label[:, 0]).permute(0, 3, 1, 2).contiguous()  # full sampled image [bs, 2, x, y]
        mask_under = data_batch[1].to(rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        mask_net_up = data_batch[2].to(rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        mask_net_down = data_batch[3].to(rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

        # print(label.shape)
        # print(mask_under.shape)
        # print(mask_net_up.shape)
        # print(mask_net_down.shape)

        # plt.imsave('label.png', abs(torch.view_as_complex(label.permute(0, 2, 3, 1).contiguous()).cpu())[0])
        # plt.imsave('mask_under.png', mask_under[0, 0].cpu())
        # plt.imsave('mask_net_up.png', mask_net_up[0, 0].cpu())
        # plt.imsave('mask_net_down.png', mask_net_down[0, 0].cpu())

        under_kspace = fft2_tensor(label) * mask_under
        under_img = ifft2_tensor(under_kspace)
        net_kspace_up = under_kspace * mask_net_up
        net_img_up = ifft2_tensor(net_kspace_up)
        net_kspace_down = under_kspace * mask_net_down
        net_img_down = ifft2_tensor(net_kspace_down)

        # plt.imsave('under_kspace.png', abs(torch.view_as_complex(under_kspace.permute(0, 2, 3, 1).contiguous()).cpu())[0])
        # plt.imsave('under_img.png', abs(torch.view_as_complex(under_img.permute(0, 2, 3, 1).contiguous()).cpu())[0])
        # plt.imsave('net_kspace_up.png', abs(torch.view_as_complex(net_kspace_up.permute(0, 2, 3, 1).contiguous()).cpu())[0])
        # plt.imsave('net_img_down.png', abs(torch.view_as_complex(net_img_down.permute(0, 2, 3, 1).contiguous()).cpu())[0])

        if mode == 'test':
            net_img_up = net_img_down = under_img
            net_kspace_up = net_kspace_down = under_kspace
            mask_net_up = mask_net_down = mask_under

        #  1
        output_k, recon_loss_up, output_i, recon_loss_down = model(mask_under,
                                                                 under_kspace,
                                                                 under_img,
                                                                 mask_net_up,
                                                                 net_kspace_up,
                                                                 mask_net_down,
                                                                 net_img_down)
        diff_otherf = (output_k - fft2_tensor(output_i)) * (1 - mask_under)
        diff_loss = criterion(diff_otherf, torch.zeros_like(diff_otherf))
        batch_loss = recon_loss_up + recon_loss_down + 0.01 * diff_loss
        # batch_loss = recon_loss_up + recon_loss_down

        # 2
        # output_up, recon_loss_up, output_down, recon_loss_down = model(mask_under,
        #                                                          under_kspace,
        #                                                          under_img,
        #                                                          mask_net_up,
        #                                                          net_img_up,
        #                                                          net_kspace_up,
        #                                                          mask_net_down,
        #                                                          net_img_down,
        #                                                          net_kspace_down)
        # diff_otherf = (fft2_tensor(output_up) - fft2_tensor(output_down)) * (1 - mask_under)
        # diff_loss = criterion(diff_otherf, torch.zeros_like(diff_otherf))
        # batch_loss = recon_loss_up + recon_loss_down + 0.01 * diff_loss
        # output_i = output_down

        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            label = torch.abs(torch.view_as_complex(label.permute(0, 2, 3, 1).contiguous()))
            output_i = torch.abs(torch.view_as_complex(output_i.permute(0, 2, 3, 1).contiguous()))
            psnr += psnr_slice(label, output_i)
            ssim += ssim_slice(label, output_i)
        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
    return log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger(args)
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim = 0.0
    # model
    model = Network(args)
    # model = IINetwork(args)
    # whether load checkpoint
    if args.pretrained or args.mode == 'test':
        model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        args.lr = lr
        best_ssim = checkpoint['best_ssim']
        model.load_state_dict(checkpoint['model'])
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim))
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        pass
        # init_weights(model, init_type=args.init_type, gain=args.gain)
        # if rank == 0:
        #     logger.info('Initialize model with {}.'.format(args.init_type))
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # criterion, optimizer, learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if not args.pretrained:
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
        scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=20)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)

    train_set, val_set, test_set = get_dataset(args)

    # test step
    if args.mode == 'test':
        # test_set = Dataset(args.test_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.test_sample_rate)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        if rank == 0:
            logger.info('The size of test dataset is {}.'.format(len(test_set)))
            logger.info('Now testing {}.'.format(args.exp_name))
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            test_log = forward('test', rank, model, test_loader, criterion, optimizer, test_log, args)
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr = test_log[1]
        test_ssim = test_log[2]
        if rank == 0:
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
        return

    # training step
    # train_set = Dataset(args.train_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.train_sample_rate)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler
    )
    # val_set = Dataset(args.val_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.val_sample_rate)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    if rank == 0:
        logger.info('The size of training dataset and validation dataset is {} and {}, respectively.'.format(len(train_set), len(val_set)))
        logger.info('Now training {}.'.format(args.exp_name))
        writer = SummaryWriter(args.loss_curve_path)
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_log = [epoch]
        epoch_start_time = time.time()

        model.train()
        train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args)

        model.eval()
        with torch.no_grad():
            train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args)

        epoch_time = time.time() - epoch_start_time
        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = train_log[2]
        val_loss = train_log[3]
        val_psnr = train_log[4]
        val_ssim = train_log[5]

        is_best = val_ssim > best_ssim
        best_ssim = max(val_ssim, best_ssim)
        if rank == 0:
            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, val_loss, val_psnr, val_ssim))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # save checkpoint
            checkpoint = {
                'epoch': epoch,
                'lr': lr,
                'best_ssim': best_ssim,
                'model': model.module.state_dict()
            }
            if not os.path.exists(args.model_save_path):
                os.makedirs(args.model_save_path)
            model_path = os.path.join(args.model_save_path, 'checkpoint.pth.tar')
            best_model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
            if is_best:
                shutil.copy(model_path, best_model_path)
        # scheduler
        if epoch <= args.warmup_epochs and not args.pretrained:
            scheduler_wu.step()
        scheduler_re.step(val_ssim)
        early_stopping(val_ssim, loss=False)
        if early_stopping.early_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
            break
    if rank == 0:
        writer.close()
    return


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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))


if __name__ == '__main__':
    main()

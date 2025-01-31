# System / Python
import os
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
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
from net import ParallelDuDoRNetwork as Network
from IXI_dataset import IXIData as Dataset
from mri_tools import rA, rAtA, rfft2
from utils import psnr_slice, ssim_slice

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

parser = argparse.ArgumentParser()

# model name
parser.add_argument('--experiment_name', type=str, default='train_DRDN_1Do_1pT1', help='give a experiment name before training')
parser.add_argument('--model_type', type=str, default='model_recurrent_dual', help='model type')   # model_recurrent_dual / model_recurrent_single / model_cascade / model_cnn
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='../Data/PROC/', help='data root folder')
parser.add_argument('--protocol_ref', type=str, default='T1', help='prior modality dataset name')   # T1 / T2 / FLAIR
parser.add_argument('--protocol_tag', type=str, default='T2', help='recon modality dataset name')   # T1 / T2 / FLAIR
parser.add_argument('--dataset', type=str, default='Cartesian', help='dataset name')   # Cartesian / Radial / Spiral

# model architectures
parser.add_argument('--net_G', type=str, default='DRDN', help='generator network')   # DRDN / SCNN
parser.add_argument('--n_recurrent', type=int, default=5, help='Number of reccurent block in model')
parser.add_argument('--use_prior', default=False, action='store_true', help='use prior')   # True / False

# loss options
parser.add_argument('--wr_L1', type=float, default=1, help='weight for reconstruction L1 loss')

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=4, help='save evaluation for every number of epochs')

parser.add_argument('--center_fractions', type=float, default=1.0/8.0, help='Cartesian: cernter fraction')
parser.add_argument('--accelerations', type=float, default=5.0, help='Cartesian: acceleration rate')

parser.add_argument('--n_lines', type=float, default=np.round(256 * 0.16), help='Radial: number of radial lines')

parser.add_argument('--n_interleaves', type=float, default=11, help='Spiral: number of interleaves')

# optimizer
# parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=10, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')



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
parser.add_argument('--batch-size', type=int, default=4, help='batch size of single gpu')
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
parser.add_argument('--model-save-path', type=str, default='./run/checkpoints/', help='save path of trained model')
parser.add_argument('--loss-curve-path', type=str, default='./run/loss_curve/', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='./run/log.txt', mode='a+', encoding='utf-8')
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


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args):
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        label = data_batch[0].to(rank, non_blocking=True)
        mask_under = data_batch[1].to(rank, non_blocking=True)
        mask_k = data_batch[2].to(rank, non_blocking=True)
        mask_i = data_batch[3].to(rank, non_blocking=True)

        
        under_img = rAtA(label, mask_under)
        under_img = torch.cat([under_img, under_img], dim=-1)
        under_kspace = rA(label, mask_under)
        k_input = rA(label, mask_k)
        i_input = rAtA(label, mask_i)
        i_input = torch.cat([i_input, i_input], dim=-1)
        if mode == 'test':
            k_input = under_kspace
            i_input = under_img
            mask_k = mask_i = mask_under
        
        label = torch.cat([label, label], dim=-1)
        label = label.movedim(-1, 1)

        mask_under = mask_under.movedim(-1, 1)
        under_kspace = under_kspace.movedim(-1, 1)
        under_img = under_img.movedim(-1, 1)
        mask_k = mask_k.movedim(-1, 1)
        k_input = k_input.movedim(-1, 1)
        mask_i = mask_i.movedim(-1, 1)
        i_input = i_input.movedim(-1, 1)

        # TODO : 输入：欠采样图像，blabla；输出：输出的图像，以及损失函数
        k_output, i_output, loss = model.forward(mask_under, under_kspace, under_img, mask_k, k_input, mask_i, i_input)

        if mode == 'train':
            model.module.backward()
        else:
            a = label.permute(0, 2, 3, 1)
            a = abs(a[..., 0] + a[..., 1] * 1j)
            b = i_output.permute(0, 2, 3, 1)
            b = abs(b[..., 0] + b[..., 1] * 1j)
            psnr += psnr_slice(a, b)
            ssim += ssim_slice(a, b)
        loss += loss.item()
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
        logger = create_logger()
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim = 0.0
    # model
    model = Network(args)
    model.setgpu([rank])
    # 统计模型参数
    if rank == 0:
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Number of parameters: {} \n'.format(num_param))
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
        # model.initialize()
        pass
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

    # test step
    if args.mode == 'test':
        test_set = Dataset(args.test_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.test_sample_rate)
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
    train_set = Dataset(args.train_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.train_sample_rate)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler
    )
    val_set = Dataset(args.val_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.val_sample_rate)
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

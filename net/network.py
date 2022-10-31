import os
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from mri_tools import *
from utils import *
from models import *


# noinspection PyAttributeOutsideInit
class ParallelKINetworkV2(nn.Module):
    def __init__(self, rank, args):
        super(ParallelKINetworkV2, self).__init__()

        self.args = args
        self.network_k = du_recurrent_model.KRNet(args)
        self.network_k.initialize()
        self.network_i = du_recurrent_model.IRNet(args)
        self.network_i.initialize()

        self.optimizer = torch.optim.Adam(list(self.network_k.parameters()) +
                                          list(self.network_i.parameters()),
                                          lr=args.lr)

        self.criterion = nn.MSELoss()

        self.epoch = 0
        self.target_metric = 'ssim2'
        self.best_target_metric = -1.

        self.save_every = 1
        self.model_dir = args.model_save_path
        self.model_path = os.path.join(self.args.model_save_path, 'checkpoint.pth.tar')
        self.best_model_path = os.path.join(self.args.model_save_path, 'best_checkpoint.pth.tar')

        self.rank = rank

        # callbacks
        self.signal_to_stop = False
        self.scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda
            epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1)
        self.scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='max', factor=0.3,
                                                                       patience=20)
        self.early_stopping = EarlyStopping(patience=50, delta=1e-5)

        self.save_test_vis = False

    def set_input_image_with_masks(self, img_full, mask_omega, mask_subset1, mask_subset2):
        """
        all shape == [bs, 2, x, y]
        """
        self.img_full = img_full
        self.k_full = fft2_tensor(img_full)

        self.k_omega = self.k_full * mask_omega
        self.img_omega = ifft2_tensor(self.k_omega)
        self.mask_omega = mask_omega

        self.k_subset1 = self.k_omega * mask_subset1
        self.img_subset1 = ifft2_tensor(self.k_subset1)
        self.mask_subset1 = mask_subset1

        self.k_subset2 = self.k_omega * mask_subset2
        self.img_subset2 = ifft2_tensor(self.k_subset2)
        self.mask_subset2 = mask_subset2

    def forward(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_subset1,
            self.k_subset1,
            self.mask_subset1,
            self.k_omega,
            self.mask_omega
        )
        output_i, loss_i_branch = self.network_i.forward(
            self.img_subset2,
            self.k_subset2,
            self.mask_subset2,
            self.k_omega,
            self.mask_omega
        )

        # some loss term based on the above (like diff loss)
        diff = (output_k - fft2_tensor(output_i)) * (1 - self.mask_omega)
        diff_loss = self.criterion(diff, torch.zeros_like(diff))
        loss = loss_k_branch + loss_i_branch + 0.01 * diff_loss

        output_i_1 = ifft2_tensor(output_k)
        output_i_2 = output_i

        return output_i_1, output_i_2, loss

    def update(self):
        output_i_1, output_i_2, loss = self.forward()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output_i_1, output_i_2, loss

    def test(self):
        output_i_1, output_i_2, loss = self.forward()

        # get magnitude images
        img_full = torch.abs(torch.view_as_complex(self.img_full.permute(0, 2, 3, 1).contiguous()))
        img_omega = torch.abs(torch.view_as_complex(self.img_omega.permute(0, 2, 3, 1).contiguous()))
        output_i_1 = torch.abs(torch.view_as_complex(output_i_1.permute(0, 2, 3, 1).contiguous()))
        output_i_2 = torch.abs(torch.view_as_complex(output_i_2.permute(0, 2, 3, 1).contiguous()))

        img_diff_2 = output_i_2 - img_full

        # calculate metrics
        psnr_1 = psnr_slice(img_full, output_i_1)
        ssim_1 = ssim_slice(img_full, output_i_1)
        psnr_2 = psnr_slice(img_full, output_i_2)
        ssim_2 = ssim_slice(img_full, output_i_2)

        if self.save_test_vis:
            if not hasattr(self, 'cnt'):
                self.cnt = 0
            else:
                self.cnt += 1
            plt.imsave(os.path.join(self.args.output_path, f'img_full_{self.cnt}.png'), img_full.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_omega_{self.cnt}.png'), img_omega.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_output1_{self.cnt}.png'), output_i_1.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_output2_{self.cnt}.png'), output_i_2.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_diff2_{self.cnt}.png'), img_diff_2.cpu()[0], cmap='bwr', vmin=-0.3, vmax=0.3)

        return output_i_1, output_i_2, loss, psnr_1, psnr_2, ssim_1, ssim_2

    def run_one_epoch(self, mode, dataloader):

        assert mode in ['train', 'val', 'test']

        tik = time.time()

        loss, psnr_1, psnr_2, ssim_1, ssim_2 = 0.0, 0.0, 0.0, 0., 0.

        for iter_num, data_batch in enumerate(dataloader):

            label = data_batch[0].to(self.rank, non_blocking=True)  # full sampled image [bs, 1, x, y]
            label = torch.view_as_real(label[:, 0]).permute(0, 3, 1, 2).contiguous()  # full sampled image [bs, 2, x, y]
            mask_under = data_batch[1].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            mask_net_up = data_batch[2].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            mask_net_down = data_batch[3].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            if mode == 'test':
                mask_net_up = mask_net_down = mask_under

            self.set_input_image_with_masks(label, mask_under, mask_net_up, mask_net_down)

            if mode == 'train':
                output_i_1, output_i_2, batch_loss = self.update()
            else:
                output_i_1, output_i_2, batch_loss, _psnr_1, _psnr_2, _ssim_1, _ssim_2 = self.test()
                psnr_1 += _psnr_1
                psnr_2 += _psnr_2
                ssim_1 += _ssim_1
                ssim_2 += _ssim_2

            loss += batch_loss.item()

        loss /= len(dataloader)

        log = dict()
        log['epoch'] = self.epoch
        log['loss'] = loss
        if mode == 'train':
            log['lr'] = self.optimizer.param_groups[0]['lr']
        else:
            psnr_1 /= len(dataloader)
            ssim_1 /= len(dataloader)
            psnr_2 /= len(dataloader)
            ssim_2 /= len(dataloader)
            log['psnr1'] = psnr_1
            log['psnr2'] = psnr_2
            log['ssim1'] = ssim_1
            log['ssim2'] = ssim_2

        tok = time.time()

        log['time'] = tok - tik

        return log

    def before_train_one_epoch(self):
        self.epoch += 1

    def train_one_epoch(self, dataloader):
        self.before_train_one_epoch()
        self.train()
        log = self.run_one_epoch('train', dataloader)
        self.after_train_one_epoch(log)
        return log

    def after_train_one_epoch(self, log):
        # warmup
        if self.epoch <= self.args.warmup_epochs:  # warmup
            self.scheduler_wu.step()

    def eval_one_epoch(self, dataloader):
        self.eval()
        with torch.no_grad():
            log = self.run_one_epoch('val', dataloader)
        self.after_eval_one_epoch(log)
        return log

    def after_eval_one_epoch(self, log):
        # update best metric
        self.best_target_metric = max(log[self.target_metric], self.best_target_metric)

        # save
        if self.epoch % self.save_every == 0:
            self.save()
        if log[self.target_metric] >= self.best_target_metric:
            self.save_best()

        # early stop
        self.early_stopping(log[self.target_metric], loss=False)
        if self.early_stopping.early_stop:
            self.signal_to_stop = True

        # reduce on plateau
        self.scheduler_re.step(log[self.target_metric])  # reduce on plateau

    def test_one_epoch(self, dataloader):
        self.eval()
        with torch.no_grad():
            log = self.run_one_epoch('test', dataloader)
        return log

    def save(self, model_path=None):
        # save checkpoint
        checkpoint = {
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_target_metric,
            'model': self.state_dict()
        }
        if model_path is None:
            model_path = self.model_path
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(checkpoint, model_path)

    def save_best(self):
        self.save(self.best_model_path)

    def load(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.rank))
        self.epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_target_metric = checkpoint['best_metric']
        self.load_state_dict(checkpoint['model'])

    def load_best(self):
        self.load(self.best_model_path)


class SemisupervisedParallelKINetworkV2(ParallelKINetworkV2):
    def run_one_epoch(self, mode, dataloader):

        assert mode in ['train', 'val', 'test']

        tik = time.time()

        loss, psnr_1, psnr_2, ssim_1, ssim_2 = 0.0, 0.0, 0.0, 0., 0.

        for iter_num, data_batch in enumerate(dataloader):

            label = data_batch[0].to(self.rank, non_blocking=True)  # full sampled image [bs, 1, x, y]
            label = torch.view_as_real(label[:, 0]).permute(0, 3, 1, 2).contiguous()  # full sampled image [bs, 2, x, y]
            mask_under = data_batch[1].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            mask_net_up = data_batch[2].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            mask_net_down = data_batch[3].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            if mode == 'train':
                # for supervised samples,
                # we should set mask_net_up and mask_net_down to mask_omega, for input and DC
                # and set mask_omega all 1, for loss calculation
                _is_unsupervised = data_batch[4]['_is_unsupervised']  # [bs, ]
                supervised_idx = torch.logical_not(_is_unsupervised).nonzero()
                mask_net_up[supervised_idx] = mask_net_down[supervised_idx] = mask_under[supervised_idx]
                mask_under[supervised_idx] = torch.ones_like(mask_under[supervised_idx])
            elif mode == 'test' or mode == 'val':
                mask_net_up = mask_net_down = mask_under

            self.set_input_image_with_masks(label, mask_under, mask_net_up, mask_net_down)

            if mode == 'train':
                output_i_1, output_i_2, batch_loss = self.update()
            else:
                output_i_1, output_i_2, batch_loss, _psnr_1, _psnr_2, _ssim_1, _ssim_2 = self.test()
                psnr_1 += _psnr_1
                psnr_2 += _psnr_2
                ssim_1 += _ssim_1
                ssim_2 += _ssim_2

            loss += batch_loss.item()

        loss /= len(dataloader)

        log = dict()
        log['epoch'] = self.epoch
        log['loss'] = loss
        if mode == 'train':
            log['lr'] = self.optimizer.param_groups[0]['lr']
        else:
            psnr_1 /= len(dataloader)
            ssim_1 /= len(dataloader)
            psnr_2 /= len(dataloader)
            ssim_2 /= len(dataloader)
            log['psnr1'] = psnr_1
            log['psnr2'] = psnr_2
            log['ssim1'] = ssim_1
            log['ssim2'] = ssim_2

        tok = time.time()

        log['time'] = tok - tik

        return log


class COSemisupervisedParallelKINetworkV2(ParallelKINetworkV2):
    """
    Dual Domain Consistency Semisupervised...
    """
    def run_one_epoch(self, mode, dataloader):

        assert mode in ['train', 'val', 'test']

        tik = time.time()

        loss, psnr_1, psnr_2, ssim_1, ssim_2 = 0.0, 0.0, 0.0, 0., 0.

        for iter_num, data_batch in enumerate(dataloader):

            label = data_batch[0].to(self.rank, non_blocking=True)  # full sampled image [bs, 1, x, y]
            label = torch.view_as_real(label[:, 0]).permute(0, 3, 1, 2).contiguous()  # full sampled image [bs, 2, x, y]
            mask_under = data_batch[1].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            mask_net_up = data_batch[2].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            mask_net_down = data_batch[3].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            if mode == 'train':
                # for supervised samples,
                # we should set mask_net_up and mask_net_down to mask_omega, for input and DC
                # and set mask_omega all 1, for loss calculation
                _is_unsupervised = data_batch[4]['_is_unsupervised']  # [bs, ]
                supervised_idx = torch.logical_not(_is_unsupervised).nonzero()
                mask_net_up[supervised_idx] = mask_net_down[supervised_idx] = mask_under[supervised_idx]
                mask_under[supervised_idx] = torch.ones_like(mask_under[supervised_idx])
                # for unsupervised samples,
                # we should set mask_net_up and mask_net_down to mask_omega, for input and DC
                # and set mask_omega to all 0, for loss calculation that only considers consistency between two branches
                # so this is called "consistency only"
                unsupervised_idx = _is_unsupervised.nonzero()
                mask_net_up[unsupervised_idx] = mask_net_down[unsupervised_idx] = mask_under[unsupervised_idx]
                mask_under[unsupervised_idx] = torch.zeros_like(mask_under[unsupervised_idx])
            elif mode == 'test' or mode == 'val':
                mask_net_up = mask_net_down = mask_under

            self.set_input_image_with_masks(label, mask_under, mask_net_up, mask_net_down)

            if mode == 'train':
                output_i_1, output_i_2, batch_loss = self.update()
            else:
                output_i_1, output_i_2, batch_loss, _psnr_1, _psnr_2, _ssim_1, _ssim_2 = self.test()
                psnr_1 += _psnr_1
                psnr_2 += _psnr_2
                ssim_1 += _ssim_1
                ssim_2 += _ssim_2

            loss += batch_loss.item()

        loss /= len(dataloader)

        log = dict()
        log['epoch'] = self.epoch
        log['loss'] = loss
        if mode == 'train':
            log['lr'] = self.optimizer.param_groups[0]['lr']
        else:
            psnr_1 /= len(dataloader)
            ssim_1 /= len(dataloader)
            psnr_2 /= len(dataloader)
            ssim_2 /= len(dataloader)
            log['psnr1'] = psnr_1
            log['psnr2'] = psnr_2
            log['ssim1'] = ssim_1
            log['ssim2'] = ssim_2

        tok = time.time()

        log['time'] = tok - tik

        return log
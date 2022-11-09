import os
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from mri_tools import *
from utils import *
from models import *
from net.base_network import BaseModel


class SemisupervisedParallelKINetworkV3(BaseModel):
    def build(self):
        self.network_k = du_recurrent_model.KRNet(self.args)
        self.network_k.initialize()
        self.network_i = du_recurrent_model.IRNet(self.args)
        self.network_i.initialize()

        self.optimizer = torch.optim.Adam(list(self.network_k.parameters()) +
                                          list(self.network_i.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()

    def set_input(self, mode, data_batch):
        """
        all shape == [bs, 2, x, y]
        """
        img_full = data_batch[0].to(self.rank, non_blocking=True)  # full sampled image [bs, 1, x, y]
        img_full = torch.view_as_real(img_full[:, 0]).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
        mask_omega = data_batch[1].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        mask_subset1 = data_batch[2].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        mask_subset2 = data_batch[3].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

        if mode == 'train':  # save unsup and sup sample idx, because they have different behaviours when forward
            _is_unsupervised = data_batch[4]['_is_unsupervised']  # [bs, ]
            unsupervised_idx = _is_unsupervised.nonzero()[:, 0]
            supervised_idx = torch.logical_not(_is_unsupervised).nonzero()[:, 0]
            self.supervised_idx = supervised_idx
            self.unsupervised_idx = unsupervised_idx

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

    def sup_train_forward(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.mask_omega[self.supervised_idx])
        )
        output_i, loss_i_branch = self.network_i.forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.mask_omega[self.supervised_idx])
        )

        loss_sup = loss_k_branch + loss_i_branch

        return loss_sup

    def unsup_train_forward(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_subset1[self.unsupervised_idx],
            self.k_subset1[self.unsupervised_idx],
            self.mask_subset1[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )
        output_i, loss_i_branch = self.network_i.forward(
            self.img_subset2[self.unsupervised_idx],
            self.k_subset2[self.unsupervised_idx],
            self.mask_subset2[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )

        # some loss term based on the above (like diff loss)
        diff = (output_k - fft2_tensor(output_i)) * (1 - self.mask_omega[self.unsupervised_idx])
        diff_loss = self.criterion(diff, torch.zeros_like(diff))
        loss_unsup = loss_k_branch + loss_i_branch + 0.01 * diff_loss

        return loss_unsup

    def train_forward(self):
        loss_sup, loss_unsup = 0., 0.

        # for sup samples
        if len(self.supervised_idx) > 0:
            loss_sup = self.sup_train_forward()

        # for unsup samples
        if len(self.unsupervised_idx) > 0:
            loss_unsup = self.unsup_train_forward()

        loss = loss_sup + loss_unsup

        return loss

    def update(self):
        loss = self.train_forward()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def inference(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.k_full)
        )
        output_i, loss_i_branch = self.network_i.forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.k_full)
        )

        loss = loss_k_branch + loss_i_branch

        output_i_1 = ifft2_tensor(output_k)
        output_i_2 = output_i

        return output_i_1, output_i_2, loss

    def post_evaluation(self):
        # get magnitude images
        img_full = torch.abs(torch.view_as_complex(self.img_full.permute(0, 2, 3, 1).contiguous()))
        img_omega = torch.abs(torch.view_as_complex(self.img_omega.permute(0, 2, 3, 1).contiguous()))
        output_i_1 = torch.abs(torch.view_as_complex(self.output_i_1.permute(0, 2, 3, 1).contiguous()))
        output_i_2 = torch.abs(torch.view_as_complex(self.output_i_2.permute(0, 2, 3, 1).contiguous()))

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

        return psnr_1, psnr_2, ssim_1, ssim_2

    def run_one_epoch(self, mode, dataloader):

        assert mode in ['train', 'val', 'test']

        tik = time.time()

        loss, psnr_1, psnr_2, ssim_1, ssim_2 = 0.0, 0.0, 0.0, 0., 0.

        for iter_num, data_batch in enumerate(dataloader):

            self.set_input(mode, data_batch)

            if mode == 'train':
                batch_loss = self.update()
            else:
                self.output_i_1, self.output_i_2, batch_loss = self.inference()
                _psnr_1, _psnr_2, _ssim_1, _ssim_2 = self.post_evaluation()
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


class SemisupervisedParallelKINetworkV4(SemisupervisedParallelKINetworkV3):
    """
    Compared to V3,
    (1) Use share-weight model, so one single model totally. But this model has a K-subnet and a I-subnet
    (2) Dual-domain consistency by: K-K-K-K -> I-K-I-K <- I-I-I-I
    (3) Because of (2), dual-input consistency and dual-domain consistency are decoupled
    """
    def build(self):
        hybrid_model = du_recurrent_model.HybridRecurrentModel(self.args)
        hybrid_model.initialize()
        self.network_up = hybrid_model
        self.network_down = hybrid_model

        self.optimizer = torch.optim.Adam(hybrid_model.parameters(),
                                          lr=self.args.lr)

        self.criterion = nn.L1Loss()  # TODO: here we use L1loss

    def sup_train_forward(self):
        """train_forward for sup samples"""
        output_sup, loss_sup = self.network_down.dual_space_forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.k_full[self.supervised_idx])
        )

        return loss_sup

    def unsup_train_forward(self):
        """train_forward for unsup samples"""
        # for unsup samples
        output_up, loss_omega_up, loss_ddc_up = self.network_up.hybrid_forward(
            self.img_subset1[self.unsupervised_idx],
            self.k_subset1[self.unsupervised_idx],
            self.mask_subset1[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )
        output_down, loss_omega_down, loss_ddc_down = self.network_down.hybrid_forward(
            self.img_subset2[self.unsupervised_idx],
            self.k_subset2[self.unsupervised_idx],
            self.mask_subset2[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )

        # omega loss
        loss_omega = loss_omega_up + loss_omega_down

        # dual-domain consistency loss
        loss_ddc = loss_ddc_up + loss_ddc_down

        # dual-input loss, outside mask_omega
        diff = (fft2_tensor(output_up) - fft2_tensor(output_down)) * (1 - self.mask_omega[self.unsupervised_idx])
        loss_di = self.criterion(diff, torch.zeros_like(diff))

        loss_unsup = loss_omega + 0.01 * loss_ddc + 0.01 * loss_di

        return loss_unsup

    def inference(self):
        output_i, loss_omega = self.network_down.dual_space_forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.mask_omega)
        )

        output_i_1 = output_i_2 = output_i
        loss = loss_omega

        return output_i_1, output_i_2, loss


class SemisupervisedParallelKINetworkV5(SemisupervisedParallelKINetworkV3):
    """
    Compared to V4,
    (1) Remove dual-domain consistency, thus no hybrid-forward conducted.
    """
    def build(self):
        hybrid_model = du_recurrent_model.HybridRecurrentModel(self.args)
        hybrid_model.initialize()
        self.network_up = hybrid_model
        self.network_down = hybrid_model

        self.optimizer = torch.optim.Adam(hybrid_model.parameters(),
                                          lr=self.args.lr)

        self.criterion = nn.L1Loss()  # TODO: here we use L1loss

    def sup_train_forward(self):
        """train_forward for sup samples"""
        output_sup, loss_sup = self.network_down.dual_space_forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.k_full[self.supervised_idx])
        )

        return loss_sup

    def unsup_train_forward(self):
        """train_forward for unsup samples"""
        # for unsup samples
        output_up, loss_omega_up = self.network_up.dual_space_forward(
            self.img_subset1[self.unsupervised_idx],
            self.k_subset1[self.unsupervised_idx],
            self.mask_subset1[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )
        output_down, loss_omega_down = self.network_down.dual_space_forward(
            self.img_subset2[self.unsupervised_idx],
            self.k_subset2[self.unsupervised_idx],
            self.mask_subset2[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )

        # omega loss
        loss_omega = loss_omega_up + loss_omega_down

        # dual-input loss, outside mask_omega
        diff = (fft2_tensor(output_up) - fft2_tensor(output_down)) * (1 - self.mask_omega[self.unsupervised_idx])
        loss_di = self.criterion(diff, torch.zeros_like(diff))

        loss_unsup = loss_omega + 0.01 * loss_di

        return loss_unsup

    def inference(self):
        output_i, loss_omega = self.network_down.dual_space_forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.mask_omega)
        )

        output_i_1 = output_i_2 = output_i
        loss = loss_omega

        return output_i_1, output_i_2, loss
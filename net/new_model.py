import os
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from mri_tools import *
from utils import *
from models import *
from base_model import BaseModel


class ParallelKINetwork(BaseModel):
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

        if mode == 'test' or mode == 'val':
            mask_subset1 = mask_subset2 = mask_omega

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

            self.set_input(mode, data_batch)

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


class ParallelKIKINetwork(ParallelKINetwork):
    def build(self):
        self.network_up = du_recurrent_model.KINet(self.args)
        self.network_up.initialize()
        self.network_down = du_recurrent_model.KINet(self.args)
        self.network_down.initialize()

        self.optimizer = torch.optim.Adam(list(self.network_up.parameters()) +
                                          list(self.network_down.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()

    def forward(self):
        output_up, loss_up = self.network_up.forward(
            self.img_subset1,
            self.k_subset1,
            self.mask_subset1,
            self.k_omega,
            self.mask_omega
        )
        output_down, loss_down = self.network_down.forward(
            self.img_subset2,
            self.k_subset2,
            self.mask_subset2,
            self.k_omega,
            self.mask_omega
        )
        # dc loss and dual-domain loss above

        # dual-input loss
        diff = (fft2_tensor(output_up) - fft2_tensor(output_down)) * (1 - self.mask_omega)
        diff_loss = self.criterion(diff, torch.zeros_like(diff))
        loss = loss_up + loss_down + 0.01 * diff_loss

        output_i_1 = output_up
        output_i_2 = output_down

        return output_i_1, output_i_2, loss

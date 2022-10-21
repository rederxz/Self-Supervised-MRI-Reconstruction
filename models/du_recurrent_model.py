import os
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
# from skimage.measure import compare_ssim as ssim

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, get_nonlinearity, DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2, complex_abs_eval
from mri_tools import fft2_tensor

import pdb


class RecurrentModel(nn.Module):
    """
    single domain recurrent model (i-domain or k-domain)
    """
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        # self.loss_names = []
        self.networks = []
        # self.optimizers = []

        self.n_recurrent = opts.n_recurrent

        # set default loss flags
        # loss_flags = ("w_img_L1")
        # for flag in loss_flags:
        #     if not hasattr(opts, flag): setattr(opts, flag, 0)

        # self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)
        #
        # if self.is_train:
        #     self.loss_names += ['loss_G_L1']
        #     param = list(self.net_G.parameters())
        #     self.optimizer_G = torch.optim.Adam(param,
        #                                         lr=opts.lr,
        #                                         betas=(opts.beta1, opts.beta2),
        #                                         weight_decay=opts.weight_decay)
        #     self.optimizers.append(self.optimizer_G)
        #
        self.criterion = nn.L1Loss()
        #
        # self.opts = opts
        #
        # # data consistency layers in image space & k-space
        # dcs_I = []
        # for i in range(self.n_recurrent):
        #     dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        # self.dcs_I = dcs_I
        #
        # dcs_K = []
        # for i in range(self.n_recurrent):
        #     dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        # self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def forward_k(self, k_input, k_target, k_target_mask):
        K = k_input
        K.requires_grad_(True)

        net = {}
        for i in range(1, self.n_recurrent + 1):

            net['r%d_kspc_pred' % i] = self.net_G(K)  # output recon kspace
            K = net['r%d_kspc_pred' % i]

        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.criterion(net['r%d_kspc_pred' % j] * k_target_mask, k_target)

        return K, loss_kspc

    def forward_i(self, i_input, i_target, i_target_mask):
        I = i_input
        I.requires_grad_(True)

        net_i = {}
        for i in range(1, self.n_recurrent + 1):

            net_i['r%d_img_pred' % i] = self.net_G(I)  # output recon image
            I = net_i['r%d_img_pred' % i]

        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(
                fft2_tensor(net_i['r%d_img_pred' % j]) * i_target_mask,
                i_target)

        return I, loss_img_dc

    def forward(self):
        I = self.tag_image_sub
        I.requires_grad_(True)

        net = {}
        for i in range(1, self.n_recurrent + 1):
            '''Image Space'''
            x_I = I

            net['r%d_img_pred' % i] = self.net_G_I(x_I)  # output recon image
            net['r%d_img_dc_pred' % i], _ = self.dcs_I[i - 1](net['r%d_img_pred' % i], self.tag_kspace_full, self.tag_kspace_mask2d)

            '''K Space'''
            net['r%d_kspc_img_dc_pred' % i] = fft2(net['r%d_img_dc_pred' % i].permute(0, 2, 3, 1))  # output data consistency image's kspace

            x_K = net['r%d_kspc_img_dc_pred' % i].permute(0, 3, 1, 2)

            net['r%d_kspc_pred' % i] = self.net_G_K(x_K)  # output recon kspace
            I, _ = self.dcs_K[i - 1](net['r%d_kspc_pred' % i], self.tag_kspace_full, self.tag_kspace_mask2d)  # output data consistency images

        self.net = net
        self.recon = I

    def update_G(self):
        loss_G_L1 = 0
        self.optimizer_G.zero_grad()

        # TODO: 这里是每次迭代的输出都参与计算loss，而不只有最终的输出，即deep supervision

        # Image domain
        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(self.net['r%d_img_dc_pred' % j], self.tag_image_full)

        # Kspace domain
        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.criterion(self.net['r%d_kspc_pred' % j].permute(0, 2, 3, 1), self.tag_kspace_full)

        loss_G_L1 = loss_img_dc + loss_kspc
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img = loss_img_dc.item()
        self.loss_kspc = loss_kspc.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def update_G(self):
        loss_G_L1 = 0
        self.optimizer_G.zero_grad()

        # TODO: 这里是每次迭代的输出都参与计算loss，而不只有最终的输出，即deep supervision

        # Image domain
        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(self.net['r%d_img_dc_pred' % j], self.tag_image_full)

        # Kspace domain
        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.criterion(self.net['r%d_kspc_pred' % j].permute(0, 2, 3, 1), self.tag_kspace_full)

        loss_G_L1 = loss_img_dc + loss_kspc
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img = loss_img_dc.item()
        self.loss_kspc = loss_kspc.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()


class RecurrentModelDC(nn.Module):
    """
    single domain recurrent model (i-domain or k-domain) with DC
    """
    def __init__(self, opts):
        super(RecurrentModelDC, self).__init__()

        self.networks = []

        self.n_recurrent = opts.n_recurrent

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        self.criterion = nn.L1Loss()

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def forward_k(self, k_input, mask_input, k_target, k_target_mask):
        K = k_input
        K.requires_grad_(True)

        net = {}
        for i in range(1, self.n_recurrent + 1):

            net['r%d_kspc_pred' % i] = self.net_G(K)  # output recon kspace
            _, K = self.dcs_K[i - 1](net['r%d_kspc_pred' % i], k_input, mask_input)  # output data consistency images

        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.criterion(net['r%d_kspc_pred' % j] * k_target_mask, k_target)

        return K, loss_kspc

    def forward_i(self, i_input, mask_input, i_target, i_target_mask):
        k_input = fft2_tensor(i_input)
        I = i_input
        I.requires_grad_(True)

        net_i = {}
        for i in range(1, self.n_recurrent + 1):

            net_i['r%d_img_pred' % i] = self.net_G(I)  # output recon image
            net_i['r%d_img_dc_pred' % i], _ = self.dcs_I[i - 1](net_i['r%d_img_pred' % i], k_input, mask_input)
            I = net_i['r%d_img_dc_pred' % i]

        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(
                fft2_tensor(net_i['r%d_img_dc_pred' % j]) * i_target_mask,
                i_target)

        return I, loss_img_dc


# noinspection PyCallingNonCallable
class KRNet(nn.Module):
    """
    single domain recurrent model (i-domain or k-domain) with DC
    """
    def __init__(self, opts):
        super(KRNet, self).__init__()

        self.networks = []

        self.n_recurrent = opts.n_recurrent

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        self.criterion = nn.L1Loss()

        # data consistency layers in image space & k-space
        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def forward_k(self, img_subset, k_subset, mask_subset, k_omega, mask_omega):
        K = k_subset
        K.requires_grad_(True)

        net = {}
        for i in range(1, self.n_recurrent + 1):

            net['r%d_kspc_pred' % i] = self.net_G(K)  # output recon kspace
            _, K = self.dcs_K[i - 1](net['r%d_kspc_pred' % i], k_subset, mask_subset)  # output data consistency images

        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.criterion(net['r%d_kspc_pred' % j] * mask_omega, k_omega)

        return K, loss_kspc

    def forward(self, *args, **kwargs):
        return self.forward_k(self, *args, **kwargs)


# noinspection PyCallingNonCallable
class IRNet(nn.Module):
    """
    single domain recurrent model (i-domain or k-domain) with DC
    """
    def __init__(self, opts):
        super(IRNet, self).__init__()

        self.networks = []

        self.n_recurrent = opts.n_recurrent

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        self.criterion = nn.L1Loss()

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def forward_i(self, img_subset, k_subset, mask_subset, k_omega, mask_omega):
        I = img_subset
        I.requires_grad_(True)

        net_i = {}
        for i in range(1, self.n_recurrent + 1):

            net_i['r%d_img_pred' % i] = self.net_G(I)  # output recon image
            net_i['r%d_img_dc_pred' % i], _ = self.dcs_I[i - 1](net_i['r%d_img_pred' % i], k_subset, mask_subset)
            I = net_i['r%d_img_dc_pred' % i]

        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(
                fft2_tensor(net_i['r%d_img_dc_pred' % j]) * mask_omega,
                k_omega)

        return I, loss_img_dc

    def forward(self, *args, **kwargs):
        return self.forward_i(self, *args, **kwargs)


class OriginalRecurrentModel(nn.Module):
    def __init__(self, opts):
        super(OriginalRecurrentModel, self).__init__()

        self.networks = []

        self.n_recurrent = opts.n_recurrent

        self.net_G_I = get_generator(opts.net_G, opts)
        self.net_G_K = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G_I)
        self.networks.append(self.net_G_K)

        # param = list(self.net_G_I.parameters()) + list(self.net_G_K.parameters())
        # self.optimizer_G = torch.optim.Adam(param,
        #                                     lr=opts.lr,
        #                                     betas=(opts.beta1, opts.beta2),
        #                                     weight_decay=opts.weight_decay)
        # self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()

        self.opts = opts

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def forward(self, input_image, input_k, input_mask, omega_k, omega_mask):
        I = input_image
        I.requires_grad_(True)

        net = {}
        for i in range(1, self.n_recurrent + 1):
            '''Image Space'''
            x_I = I

            net['r%d_img_pred' % i] = self.net_G_I(x_I)  # output recon image
            net['r%d_img_dc_pred' % i], _ = self.dcs_I[i - 1](net['r%d_img_pred' % i], input_k, input_mask)

            '''K Space'''
            net['r%d_kspc_img_dc_pred' % i] = fft2_tensor(net['r%d_img_dc_pred' % i])  # output data consistency image's kspace

            x_K = net['r%d_kspc_img_dc_pred' % i]

            net['r%d_kspc_pred' % i] = self.net_G_K(x_K)  # output recon kspace
            I, _ = self.dcs_K[i - 1](net['r%d_kspc_pred' % i], input_k, input_mask)  # output data consistency images

        self.net = net
        self.recon = I

        # calculate the loss
        loss_G_L1 = 0

        # Image domain
        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(fft2_tensor(self.net['r%d_img_dc_pred' % j]) * omega_mask, omega_k)

        # Kspace domain
        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.criterion(self.net['r%d_kspc_pred' % j] * omega_mask, omega_k)

        loss_G_L1 = loss_img_dc + loss_kspc
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img = loss_img_dc.item()
        self.loss_kspc = loss_kspc.item()

        total_loss = loss_G_L1

        return I, total_loss

    # def update_G(self):
    #     loss_G_L1 = 0
    #     self.optimizer_G.zero_grad()
    #
    #     # Image domain
    #     loss_img_dc = 0
    #     for j in range(1, self.n_recurrent + 1):
    #         loss_img_dc = loss_img_dc + self.criterion(self.net['r%d_img_dc_pred' % j], self.tag_image_full)
    #
    #     # Kspace domain
    #     loss_kspc = 0
    #     for j in range(1, self.n_recurrent + 1):
    #         loss_kspc = loss_kspc + self.criterion(self.net['r%d_kspc_pred' % j].permute(0, 2, 3, 1), self.tag_kspace_full)
    #
    #     loss_G_L1 = loss_img_dc + loss_kspc
    #     self.loss_G_L1 = loss_G_L1.item()
    #     self.loss_img = loss_img_dc.item()
    #     self.loss_kspc = loss_kspc.item()
    #
    #     total_loss = loss_G_L1
    #     total_loss.backward()
    #     self.optimizer_G.step()
    #
    # def optimize(self):
    #     self.loss_G_L1 = 0
    #
    #     self.forward()
    #     self.update_G()

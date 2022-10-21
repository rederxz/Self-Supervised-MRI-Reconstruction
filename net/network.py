import os

import torch
import torch.optim as optim
from tqdm import tqdm

from net.net_parts import *
from mri_tools import *
from utils import *
from models import *

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


class ISTANetPlus(nn.Module):
    def __init__(self, num_layers, rank):
        super(ISTANetPlus, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(BasicBlock(self.rank))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, under_img, mask):
        x = under_img
        layers_sym = []
        for i in range(self.num_layers):
            [x, layer_sym] = self.layers[i](x, under_img, mask)
            layers_sym.append(layer_sym)
        x_final = x
        return [x_final, layers_sym]


class ParallelNetwork(nn.Module):
    def __init__(self, num_layers, rank):
        super(ParallelNetwork, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.up_network = ISTANetPlus(self.num_layers, self.rank)
        self.down_network = ISTANetPlus(self.num_layers, self.rank)

    def forward(self, under_img_up, mask_up, under_img_down, mask_down):
        output_up, loss_layers_up = self.up_network(under_img_up, mask_up)
        output_down, loss_layers_down = self.down_network(under_img_down, mask_down)
        return output_up, loss_layers_up, output_down, loss_layers_down


class ParallelKINetwork(nn.Module):
    def __init__(self, opts):
        super(ParallelKINetwork, self).__init__()

        self.k_network = du_recurrent_model.RecurrentModel(opts)
        self.k_network.initialize()
        self.i_network = du_recurrent_model.RecurrentModel(opts)
        self.i_network.initialize()

    def forward(self, mask_1, mask_1_k, mask_1_i, mask_2, mask_2_k, mask_3, mask_3_i):
        k_output, loss_k_branch = self.k_network.forward_k(mask_2_k, mask_1_k, mask_1)
        i_output, loss_i_branch = self.i_network.forward_i(mask_3_i, mask_1_k, mask_1)
        return k_output, loss_k_branch, i_output, loss_i_branch


class ParallelKINetworkDC(nn.Module):
    def __init__(self, opts):
        super(ParallelKINetworkDC, self).__init__()

        self.k_network = du_recurrent_model.RecurrentModelDC(opts)
        self.k_network.initialize()
        self.i_network = du_recurrent_model.RecurrentModelDC(opts)
        self.i_network.initialize()

    def forward(self, mask_1, mask_1_k, mask_1_i, mask_2, mask_2_k, mask_3, mask_3_i):
        k_output, loss_k_branch = self.k_network.forward_k(mask_2_k, mask_2, mask_1_k, mask_1)
        i_output, loss_i_branch = self.i_network.forward_i(mask_3_i, mask_3, mask_1_k, mask_1)
        return k_output, loss_k_branch, i_output, loss_i_branch


class ParallelDuDoRNetwork(nn.Module):
    def __init__(self, opts):
        super(ParallelDuDoRNetwork, self).__init__()

        self.up_network = du_recurrent_model.OriginalRecurrentModel(opts)
        self.up_network.initialize()
        # self.down_network = du_recurrent_model.OriginalRecurrentModel(opts)
        # self.down_network.initialize()
        self.down_network = self.up_network

    def forward(self, omega_mask, omega_k, omega_i, mask_up, i_up, k_up, mask_down, i_down, k_down):
        output_up, loss_up = self.up_network.forward(i_up, k_up, mask_up, omega_k, omega_mask)
        output_down, loss_down = self.down_network.forward(i_down, k_down, mask_down, omega_k, omega_mask)
        return output_up, loss_up, output_down, loss_down


# noinspection PyAttributeOutsideInit
class ParallelKINetworkV2(nn.Module):
    def __init__(self, opts):
        super(ParallelKINetworkV2, self).__init__()

        self.network_1 = du_recurrent_model.KRNet(opts)
        self.network_1.initialize()
        self.network_2 = du_recurrent_model.IRNet(opts)
        self.network_2.initialize()

        self.criterion = nn.L1Loss()

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

        # network_1_input = [...]  # or dict
        # network_2_input = [...]  # or dict
        # self.network_1.set_input(*network_1_input)
        # self.network_2.set_input(*network_2_input)

    def forward(self):
        output_i, loss_1 = self.network_1.forward(
            self.img_subset1,
            self.k_subset1,
            self.mask_subset1,
            self.k_omega,
            self.mask_omega
        )
        output_k, loss_2 = self.network_2.forward(
            self.img_subset2,
            self.k_subset2,
            self.mask_subset2,
            self.k_omega,
            self.mask_omega
        )

        # some loss term based on the above (like diff loss)

        diff = (output_k - fft2_tensor(output_i)) * (1 - self.mask_omega)
        diff_loss = self.criterion(diff, torch.zeros_like(diff))
        loss = loss_1 + loss_2 + 0.01 * diff_loss

        return output_i, output_k, loss

    def update(self):
        pass




class ShareWeightParallelNetwork(nn.Module):
    def __init__(self, num_layers, rank):
        super(ShareWeightParallelNetwork, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.up_network = ISTANetPlus(self.num_layers, self.rank)
        self.down_network = self.up_network

    def forward(self, under_img_up, mask_up, under_img_down, mask_down):
        output_up, loss_layers_up = self.up_network(under_img_up, mask_up)
        output_down, loss_layers_down = self.down_network(under_img_down, mask_down)
        return output_up, loss_layers_up, output_down, loss_layers_down

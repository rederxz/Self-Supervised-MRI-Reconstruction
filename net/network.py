import os

import torch
import torch.optim as optim
from tqdm import tqdm

from net.net_parts import *
from mri_tools import *
from utils import *
from models import du_recurrent_model

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

class ParallelDuDoRNetwork(nn.Module):
    def __init__(self, opts):
        super(ParallelDuDoRNetwork, self).__init__()

        self.k_network = du_recurrent_model.RecurrentModel(opts)
        self.i_network = du_recurrent_model.RecurrentModel(opts)

        self.optimizer = torch.optim.Adam(list(self.k_network.parameters()) + list(self.i_network.parameters()),
                                        lr=opts.lr,
                                        betas=(opts.beta1, opts.beta2),
                                        weight_decay=opts.weight_decay)

        self.criterion = nn.L1Loss()

    def forward(self, mask_1, mask_1_k, mask_1_i, mask_2, mask_2_k, mask_3, mask_3_i):
        self.loss = 0
        k_output, k_loss = self.k_network.forward_k(mask_2_k, mask_1_k, mask_1)
        i_output, i_loss = self.i_network.forward_i(mask_3_i, mask_1_i, mask_1)

        diff_loss = (k_output - rfft2(i_output)) * (1 - mask_1)

        self.loss = k_loss + i_loss + 0.01 * diff_loss

        return k_output, i_output, self.loss

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def setgpu(self, gpu_ids):
        self.device = gpu_ids[0]

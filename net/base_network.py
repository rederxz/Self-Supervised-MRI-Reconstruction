import os
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from mri_tools import *
from utils import *
from models import *


class BaseModel(nn.Module):
    def __init__(self, rank, args):
        super(BaseModel, self).__init__()

        self.args = args
        self.rank = rank

        self.epoch = 0
        self.target_metric = 'ssim2'
        self.best_target_metric = -1.

        self.save_every = 1
        self.model_dir = args.model_save_path
        self.model_path = os.path.join(self.args.model_save_path, 'checkpoint.pth.tar')
        self.best_model_path = os.path.join(self.args.model_save_path, 'best_checkpoint.pth.tar')

        self.build()

        self.attach_callbacks()

        self.save_test_vis = False

    def build(self):
        """model, optimizer, criterion"""
        raise NotImplementedError

    def attach_callbacks(self):
        # callbacks
        self.signal_to_stop = False
        self.scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda
            epoch: epoch / self.args.warmup_epochs if epoch <= self.args.warmup_epochs else 1)
        self.scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='max', factor=0.3,
                                                                       patience=20)
        self.early_stopping = EarlyStopping(patience=50, delta=1e-5)

    def set_input(self, mode, batch_data):
        """
        all shape == [bs, 2, x, y]
        """
        raise NotImplementedError

    def forward(self):
        """model forward and compute loss"""
        raise NotImplementedError

    def train_forward(self):
        """model forward when training"""
        raise NotImplementedError

    def update(self):
        """model forward, compute loss and model backward"""
        raise NotImplementedError

    def test(self):
        """model forward, compute loss, calculate metrics (and output viz)"""
        raise NotImplementedError

    def inference(self):
        """model inference, compute loss (save some necessary calculations)"""
        raise NotImplementedError

    def post_evaluation(self):
        raise NotImplementedError

    def run_one_epoch(self, mode, dataloader):
        """run one epoch, in 'train', 'val' or 'test' mode"""
        raise NotImplementedError

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


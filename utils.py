import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import logging
from tqdm import tqdm
import numpy as np
from paired_dataset import get_paired_volume_datasets


def psnr_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    PSNR = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        PSNR += peak_signal_noise_ratio(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)
    return PSNR / batch_size


def ssim_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    SSIM = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        SSIM += structural_similarity(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)
    return SSIM / batch_size


def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def normalize_zero_to_one(data, eps=0.):
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)



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


class SaveBest:
    def __init__(self, model):
        self.model = model
        self.best_score = model.best_score

    def __call__(self, score, higher=True):
        """higher is better"""
        better = score > self.best_score if higher else score < self.best_score
        if better:
            self.model.best_score = score
            self.model.save()
            self.model.savebest()


class Prefetch(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = [i for i in tqdm(dataset, leave=False)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind]


def get_dataset(args):
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

    return slices_train, slices_val, slices_test


def get_dataset_split(args, split):
    volumes = get_paired_volume_datasets(
            getattr(args, split), crop=256, protocals=['T2'],
            object_limit=getattr(args, split+'_obj_limit'),
            u_mask_path=args.u_mask_path,
            s_mask_up_path=args.s_mask_up_path,
            s_mask_down_path=args.s_mask_down_path)
    slices = torch.utils.data.ConcatDataset(volumes)
    if args.prefetch:
        # load all data to ram
        slices = Prefetch(slices)

    return slices


class SemisupervisedConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, supervised_every, datasets):
        """
        supervised_every: one supervised volume every how many unsupervised volumes
            like supervised_every == 3, then 1 sup / 3 unsup
        """
        super(SemisupervisedConcatDataset, self).__init__(datasets)

        supervised_idx = list()
        supervised_volume_idx = list()
        unsupervised_idx = list()
        unsupervised_volume_idx = list()
        cur_start_idx = 0
        for i, dataset in enumerate(datasets):
            cur_end_idx = cur_start_idx + len(dataset)
            if (i+1) % supervised_every == 0:
                supervised_idx += list(range(cur_start_idx, cur_end_idx))
                supervised_volume_idx.append(i)
            else:
                unsupervised_idx += list(range(cur_start_idx, cur_end_idx))
                unsupervised_volume_idx.append(i)
            cur_start_idx += len(dataset)

        self.supervised_idx = supervised_idx
        self.unsupervised_idx = unsupervised_idx
        self.supervised_volume_idx = supervised_volume_idx
        self.unsupervised_volume_idx = unsupervised_volume_idx

        # print('sup volume idx', supervised_volume_idx)
        # print('unsup volume idx', unsupervised_volume_idx)
        # print('sup slice idx', self.supervised_idx)
        # print('unsup slice idx', self.unsupervised_idx)

        assert len(np.intersect1d(self.supervised_idx, self.unsupervised_idx)) == 0
        assert np.array_equal(np.sort(np.union1d(self.supervised_idx, self.unsupervised_idx)), np.arange(0, len(self)))


    def get_supervised_idxs(self):
        return self.supervised_idx

    def get_unsupervised_idxs(self):
        return self.unsupervised_idx


def get_semisupervised_dataset_split(args, split, supervised_every):
    volumes = get_paired_volume_datasets(
        getattr(args, split), crop=256, protocals=['T2'],
        object_limit=getattr(args, split+'_obj_limit'),
        u_mask_path=args.u_mask_path,
        s_mask_up_path=args.s_mask_up_path,
        s_mask_down_path=args.s_mask_down_path,
        supervised_every=supervised_every)

    slices = SemisupervisedConcatDataset(supervised_every, volumes)
    # if args.prefetch:
    #     # load all data to ram
    #     slices = Prefetch(slices)

    return slices








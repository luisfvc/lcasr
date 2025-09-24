import argparse
from functools import partial
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Sampler

from lcasr.utils.utils import load_yaml, set_remote_paths
from lcasr.msmd_dataset import load_system_dataset
from lcasr.snippet_dataset import load_msmd_dataset
from lcasr.models.vgg_model import CrossModalEncoder, CMSnippetEncoder


class CustomSampler(Sampler):
    """ Custom sampler that samples a new subset every epoch.
        Reference: https://discuss.pytorch.org/t/new-subset-every-epoch/85018 """

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


class CustomBatch:
    def __init__(self, batch):
        self.scores = [torch.as_tensor(x[0], dtype=torch.float32).unsqueeze(1) for x in batch]
        self.specs = [torch.as_tensor(x[1], dtype=torch.float32).unsqueeze(1) for x in batch]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.scores = [s.pin_memory() for s in self.scores]
        self.specs = [s.pin_memory() for s in self.specs]
        return self


def collate_wrapper(batch):
    return CustomBatch(batch)


def create_msmd_train_loaders(args, only_refine=False):
    print('Loading and preparing data...\n')

    splits = load_yaml(os.path.join(args.split_root, f'msmd_split.yaml'))

    tr_dataset = load_system_dataset(args.msmd_root, splits['train'], args, aug='full_aug')

    if only_refine:
        # todo: check if it makes sense to have n_refine parameter
        tr_samples = random.sample(range(len(tr_dataset)), k=args.n_refine)
        tr_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_samples),
                               batch_size=args.batch_size,
                               shuffle=False,
                               drop_last=False,
                               num_workers=args.n_workers)
        return tr_loader
    print(f'Training dataset: {len(tr_dataset)} snippet pairs')

    va_dataset = load_system_dataset(args.msmd_root, splits['valid'], args, aug='no_aug')
    print(f'Validation dataset: {len(va_dataset)} snippet pairs\n')

    n_train = min([args.n_train, len(tr_dataset)])
    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.n_workers,
                           drop_last=True,
                           sampler=CustomSampler(data_source=tr_dataset, num_samples=n_train),
                           collate_fn=collate_wrapper,
                           pin_memory=True)

    n_valid = np.min([args.n_valid, len(va_dataset)])
    va_samples = np.linspace(0, len(va_dataset) - 1, n_valid).astype(int)

    va_loader = DataLoader(dataset=Subset(va_dataset, indices=va_samples), batch_size=args.batch_size,
                           num_workers=args.n_workers, drop_last=False, shuffle=False, collate_fn=collate_wrapper,
                           pin_memory=True)

    # additional dataloader for evaluating on a subset of the train set
    tr_eval_samples = np.linspace(0, len(tr_dataset) - 1, n_valid).astype(int)
    tr_eval_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_eval_samples), batch_size=args.batch_size,
                                num_workers=args.n_workers, drop_last=False, shuffle=False, collate_fn=collate_wrapper,
                                pin_memory=True)

    return tr_loader, va_loader, tr_eval_loader


def create_snip_train_loaders(args, only_refine=False):
    print('Loading and preparing data...\n')

    splits = load_yaml(os.path.join(args.split_root, f'msmd_split.yaml'))

    tr_dataset = load_msmd_dataset(args.msmd_root, splits['train'], args, aug='full_aug')

    if only_refine:
        tr_samples = random.sample(range(len(tr_dataset)), k=args.n_refine)
        tr_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_samples),
                               batch_size=args.batch_size,
                               shuffle=False,
                               drop_last=False,
                               num_workers=args.n_workers)
        return tr_loader
    print(f'Training dataset: {len(tr_dataset)} snippet pairs')

    va_dataset = load_msmd_dataset(args.msmd_root, splits['valid'], args, aug='no_aug')
    print(f'Validation dataset: {len(va_dataset)} snippet pairs\n')

    n_valid = np.min([args.n_valid, len(va_dataset)])
    va_samples = np.linspace(0, len(va_dataset) - 1, n_valid).astype(int)

    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.n_workers,
                           drop_last=True,
                           sampler=CustomSampler(data_source=tr_dataset, num_samples=args.n_train))

    va_loader = DataLoader(dataset=Subset(va_dataset, indices=va_samples), batch_size=args.batch_size,
                           num_workers=args.n_workers, drop_last=False, shuffle=False)

    # additional dataloader for evaluating on a subset of the train set
    tr_eval_samples = np.linspace(0, len(tr_dataset) - 1, n_valid).astype(int)
    tr_eval_loader = DataLoader(dataset=Subset(tr_dataset, indices=tr_eval_samples), batch_size=args.batch_size,
                                num_workers=args.n_workers, drop_last=False, shuffle=False)

    return tr_loader, va_loader, tr_eval_loader


def create_test_loader(args):
    datasets = {'MSMD': 'msmd_split.yaml', 'RealScores_Synth': 'db_scanned_synth.yaml',
                'RealScores_Rec': 'db_scanned_recording.yaml'}

    splits = load_yaml(os.path.join(args.split_root, datasets[args.dataset]))

    if args.dataset == 'MSMD':
        te_dataset = load_system_dataset(args.msmd_root, splits['test'], args, aug='test_aug')
    else:
        te_dataset = load_umc_dataset(args.umc_root, splits['test'], args)

    print(f'Test dataset: {len(te_dataset)} snippet pairs\n')
    durations = te_dataset.get_durations()
    n_test = min([args.n_test, len(te_dataset)])
    te_samples = np.linspace(0, len(te_dataset) - 1, n_test).astype(int)
    te_loader = DataLoader(dataset=Subset(te_dataset, indices=te_samples),
                           batch_size=args.batch_size,
                           drop_last=False,
                           shuffle=False,
                           num_workers=args.n_workers,
                           collate_fn=collate_wrapper)

    return te_loader, n_test, durations


def get_model(args, snippet_model=False):

    if not snippet_model:
        from ucasr.utils.losses import triplet_loss
        loss_function = partial(triplet_loss, margin=args.loss_margin)
        return CrossModalEncoder(args, use_cca=args.finetune, pre_rnn_norm=args.pre_rnn_norm, post_rnn_norm=args.post_rnn_norm),\
            loss_function

    else:
        from ucasr.utils.losses import triplet_loss
        loss_function = partial(triplet_loss, margin=args.loss_margin)
        return CMSnippetEncoder(args, use_cca=True), loss_function


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_config', help='augmentation configuration', type=str, default='full_aug')

    configs = load_yaml('config/msmd_config.yaml')
    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    arguments = set_remote_paths(parser.parse_args())
    tr_load, va_load, tr_eval_load = create_snip_train_loaders(arguments)

# encoding: utf-8

from torch.utils import data
import torch
from torch.nn.utils.rnn import pad_sequence
from .datasets.semeval_dataset import SemevalDatset


def build_dataset(csv):
    datasets = SemevalDatset(csv)
    return datasets


def make_data_loader(cfg, csv, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE
        shuffle = True
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False

    datasets = build_dataset(csv)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    data_loader = data.DataLoader(
        dataset=datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=my_collate_fn)

    return data_loader


def my_collate_fn(batch):
    inputs, values, explanations = zip(*batch)
    inputs_padded = pad_sequence([i.clone().detach() for i in inputs], batch_first=True, padding_value=1)
    values = torch.tensor(values) if not isinstance(values[0], torch.Tensor) else torch.stack(values)
    explanations = torch.tensor(explanations) if not isinstance(explanations[0], torch.Tensor) else torch.stack(explanations)

    return inputs_padded, values, explanations

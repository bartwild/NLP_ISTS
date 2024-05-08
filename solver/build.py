# encoding: utf-8

import torch


def make_optimizer(cfg, model):
    """
    Create an optimizer for the given model.

    Args:
        cfg (Config): The configuration object.
        model (torch.nn.Module): The model to optimize.

    Returns:
        torch.optim.Optimizer: The optimizer.

    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    return optimizer
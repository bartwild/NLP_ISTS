# Import necessary modules
import argparse
import os
import sys
from os import mkdir

import torch.nn.functional as F

# Add the current directory to the system path to allow relative imports
sys.path.append('.')
from net_config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer
import logging


def train(cfg):
    """
    Trains the model using the provided configuration.

    Args:
        cfg (Config): The configuration object containing the training settings.

    Returns:
        None
    """
    # Build the model using the provided configuration
    model = build_model(cfg)
    # Create the optimizer
    optimizer = make_optimizer(cfg, model)
    # Create the data loader for the training dataset
    train_loader = make_data_loader(cfg, csv=cfg.DATASETS.TRAIN, is_train=True)
    # Perform the training loop
    do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        losses=[F.mse_loss, F.nll_loss],  # List of loss functions to use
    )


def main():
    """
    Main function for training RoBERTa model for iSTS task.

    This function parses command-line arguments, merges configuration options,
    sets up logging, and starts the training process.

    Args:
        None

    Returns:
        None
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="PyTorch training RoBERTa model for iSTS task")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    # Parse arguments
    args = parser.parse_args()

    # Determine the number of GPUs available
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    # Load configuration from file if provided
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  # Freeze the configuration to prevent further modifications

    # Create the output directory if it does not exist
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    # Set up logging
    logger = logging.getLogger("train_net")
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    logger.propagate = False

    # Log the configuration file if provided
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Start the training process
    train(cfg)


if __name__ == '__main__':
    main()

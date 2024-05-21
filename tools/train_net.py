import argparse
import os
import sys
import logging
from os import mkdir

import torch.nn.functional as F

# Add the current directory to the system path to allow relative imports
sys.path.append('.')
from net_config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer


def setup_logging(output_dir):
    """
    Sets up the logging configuration.

    Args:
        output_dir (str): The directory where logs will be saved.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger("train_net")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # If an output directory is specified, create it and set up file logging
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)
    if output_dir:
        file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False  # Prevent the logger from propagating messages to the root logger
    return logger


def load_config(args):
    """
    Loads and merges configuration from file and command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    # Merge configurations from the specified config file and command-line options
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  # Freeze the configuration to prevent further modifications


def train(cfg, logger):
    """
    Trains the model using the provided configuration.

    Args:
        cfg (Config): The configuration object containing the training settings.
        logger (logging.Logger): Logger instance for logging training progress.

    Returns:
        None
    """
    # Build the model using the provided configuration
    model = build_model(cfg)
    # Create the optimizer for training
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
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    # Parse command-line arguments
    args = parser.parse_args()

    # Load and merge configuration settings
    load_config(args)

    # Set up logging
    logger = setup_logging(cfg.OUTPUT_DIR)
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))  # Determine the number of GPUs available
    logger.info(f"Using {num_gpus} GPU(s)")
    logger.info(f"Command-line args: {args}")

    # Log the configuration file content if provided
    if args.config_file:
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")

    # Start the training process
    train(cfg, logger)


if __name__ == '__main__':
    main()

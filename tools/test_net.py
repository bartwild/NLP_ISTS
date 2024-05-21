# Import necessary modules
import argparse
import os
import sys
from os import mkdir

import torch
from torch.utils.data.dataset import Dataset

# Add the current directory to the system path to allow relative imports
sys.path.append('.')
from net_config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
import logging


def main():
    """
    Main function for running the Roberta iSTS Inference.

    Args:
        None

    Returns:
        None
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Roberta iSTS Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    # Parse arguments
    args = parser.parse_args()
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
    logger = logging.getLogger("test_net")
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

    # Build the model and load weights
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))

    # Create the data loader for the validation dataset
    val_loader = make_data_loader(cfg, cfg.DATASETS.TEST, is_train=False)

    # Perform inference using the model and validation data
    inference(cfg, model, val_loader)


if __name__ == '__main__':
    main()

import argparse
import os
import sys
import logging
import torch

# Add the current directory to the system path to allow relative imports
sys.path.append('.')
from net_config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model


def setup_logging(output_dir):
    """
    Sets up the logging configuration.

    Args:
        output_dir (str): The directory where logs will be saved.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger("test_net")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)
    if output_dir:
        file_handler = logging.FileHandler(os.path.join(output_dir, "inference.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def load_config(args):
    """
    Loads and merges configuration from file and command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  # Freeze the configuration to prevent further modifications


def perform_inference(cfg, logger):
    """
    Performs inference using the provided configuration and logs the process.

    Args:
        cfg (Config): The configuration object containing inference settings.
        logger (logging.Logger): Logger instance for logging inference progress.

    Returns:
        None
    """
    # Build the model using the provided configuration
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))

    # Create the data loader for the validation dataset
    val_loader = make_data_loader(cfg, cfg.DATASETS.TEST, is_train=False)

    # Perform inference using the model and validation data
    inference(cfg, model, val_loader)


def main():
    """
    Main function for running the Roberta iSTS Inference.

    This function parses command-line arguments, merges configuration options,
    sets up logging, and starts the inference process.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Roberta iSTS Inference")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    load_config(args)

    logger = setup_logging(cfg.OUTPUT_DIR)
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"Using {num_gpus} GPU(s)")
    logger.info(f"Command-line args: {args}")

    if args.config_file:
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")

    perform_inference(cfg, logger)


if __name__ == '__main__':
    main()

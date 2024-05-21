import logging
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def setup_device(cfg):
    """
    Sets up the device for inference.

    Args:
        cfg (Config): The configuration object containing model and inference settings.

    Returns:
        torch.device: The device to be used for inference.
    """
    return torch.device(cfg.MODEL.DEVICE)


def log_progress(i, len_val_loader, log_period, logger):
    """
    Logs the progress of the inference process.

    Args:
        i (int): Current batch number.
        len_val_loader (int): Total number of batches in the validation loader.
        log_period (int): Period for logging.
        logger (logging.Logger): Logger instance.

    Returns:
        None
    """
    if i % log_period == log_period - 1:
        logger.info('Progress [%d/%d]' % (i + 1, len_val_loader))


def log_metrics(gold_exp, pred_exp, gold_val, pred_val, logger):
    """
    Logs the performance metrics for the inference.

    Args:
        gold_exp (list): List of true explanation values.
        pred_exp (list): List of predicted explanation values.
        gold_val (list): List of true values.
        pred_val (list): List of predicted values.
        logger (logging.Logger): Logger instance.

    Returns:
        None
    """
    logger.info('| F1 score for explanations: %.3f' % f1_score(gold_exp, pred_exp, average='micro'))
    logger.info('| Pearson for explanations: {:.3f}'.format(pearsonr(gold_exp, pred_exp)[0]))
    logger.info('| F1 score for values: %.3f' % f1_score(gold_val, pred_val, average='micro'))
    logger.info('| Pearson for values: {:.3f}'.format(pearsonr(gold_val, pred_val)[0]))


def save_confusion_matrix(gold_exp, pred_exp, filename="matrix_values"):
    """
    Generates and saves the confusion matrix for the explanations.

    Args:
        gold_exp (list): List of true explanation values.
        pred_exp (list): List of predicted explanation values.
        filename (str): Filename to save the confusion matrix.

    Returns:
        None
    """
    conf_matrix = confusion_matrix(gold_exp, pred_exp)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Explanations')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.savefig(filename)


def inference(cfg, model, val_loader):
    """
    Perform inference on the given validation data using the provided model.

    Args:
        cfg (Config): The configuration object containing model and inference settings.
        model (torch.nn.Module): The model to be used for inference.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation data.

    Returns:
        None
    """
    # Set device for inference
    device = setup_device(cfg)
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("model.inference")
    logger.info("Start inferencing")

    # Initialize lists to store true and predicted values and explanations
    gold_val = []
    gold_exp = []
    pred_val = []
    pred_exp = []

    # Set the model to evaluation mode and move it to the device
    model.eval()
    model.to(device)

    # Disable gradient calculation for inference
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Move inputs, values, and explanations to the device
            inputs, values, explanations = data[0].to(device), data[1].to(device), data[2].to(device)
            # Perform model inference
            out1, out2 = model(inputs)
            _, predicted = torch.max(out2, 1)  # Get the predicted explanations
            out1 = torch.round(out1)  # Round the predicted values

            # Convert tensors to lists and extend the respective lists
            gold_val.extend(values.cpu().numpy().tolist())
            gold_exp.extend(explanations.cpu().numpy().tolist())
            pred_val.extend(out1.cpu().numpy().tolist())
            pred_exp.extend(predicted.cpu().numpy().tolist())

            # Log progress periodically
            log_progress(i, len(val_loader), log_period, logger)

    # Calculate and log performance metrics
    log_metrics(gold_exp, pred_exp, gold_val, pred_val, logger)

    # Generate and save confusion matrix for explanations
    save_confusion_matrix(gold_exp, pred_exp)

# Import necessary modules
import logging
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns


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
    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
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
            gold_val.extend(values.cpu().numpy().tolist())  # Convert tensor to list and extend gold_val
            gold_exp.extend(explanations.cpu().numpy().tolist())  # Convert tensor to list and extend gold_exp
            pred_val.extend(out1.cpu().numpy().tolist())  # Convert tensor to list and extend pred_val
            pred_exp.extend(predicted.cpu().numpy().tolist())  # Convert tensor to list and extend pred_exp

            # Log progress periodically
            if i % log_period == log_period - 1:
                logger.info('Progress [%d/%d]' % (i + 1, len(val_loader)))

    # Calculate and log performance metrics
    logger.info('| F1 score for explanations: %.3f' % f1_score(gold_exp, pred_exp, average='micro'))
    logger.info('| Pearson for explanations: {:.3f}'.format(pearsonr(gold_exp, pred_exp)[0]))
    logger.info('| F1 score for values: %.3f' % f1_score(gold_val, pred_val, average='micro'))
    logger.info('| Pearson for values: {:.3f}'.format(pearsonr(gold_val, pred_val)[0]))

    # Generate and save confusion matrix for explanations
    conf_matrix = confusion_matrix(gold_exp, pred_exp)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Explanations')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.savefig("matrix_values")

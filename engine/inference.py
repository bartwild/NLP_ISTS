import logging
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import torch

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
    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("model.inference")
    logger.info("Start inferencing")

    gold_val = []
    gold_exp = []
    pred_val = []
    pred_exp = []
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, values, explanations = data[0].to(device), data[1].to(device), data[2].to(device)
            out1, out2 = model(inputs)
            _, predicted = torch.max(out2, 1)
            gold_val.extend(values.cpu().numpy().tolist())  # Convert tensor to list and extend gold_val
            gold_exp.extend(explanations.cpu().numpy().tolist())  # Convert tensor to list and extend gold_exp
            pred_val.extend(out1.cpu().numpy().tolist())  # Convert tensor to list and extend pred_val
            pred_exp.extend(predicted.cpu().numpy().tolist())  # Convert tensor to list and extend pred_exp
            #print(gold_val, pred_val)
            #print(gold_exp, pred_exp)
            if i % log_period == log_period -1:
                logger.info('Progress [%d/%d]' % (i + 1, len(val_loader)))

    logger.info('| F1 score for explanations: %.3f' % f1_score(gold_exp, pred_exp, average='micro'))
    logger.info('| Pearson for explanations: {:.3f}'.format(pearsonr(gold_exp, pred_exp)[0]))
    round_to_whole = [round(num) for num in pred_val]
    logger.info('| F1 score for values: %.3f' % f1_score(gold_val, round_to_whole, average='micro'))
    logger.info('| Pearson for values: {:.3f}'.format(pearsonr(gold_val, pred_val)[0]))

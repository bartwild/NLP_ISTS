import logging
import torch
import matplotlib.pyplot as plt


def setup_device(cfg):
    """
    Sets up the device for training.

    Args:
        cfg (object): Configuration object containing training settings.

    Returns:
        torch.device: The device to be used for training.
    """
    device_type = cfg.MODEL.DEVICE
    return torch.device(device_type)


def log_training_info(epoch, epochs, i, len_train_loader, partial_loss1, partial_loss2, log_period, logger):
    """
    Logs training information.

    Args:
        epoch (int): Current epoch number.
        epochs (int): Total number of epochs.
        i (int): Current batch number.
        len_train_loader (int): Total number of batches in the train loader.
        partial_loss1 (float): Partial loss value 1.
        partial_loss2 (float): Partial loss value 2.
        log_period (int): Period for logging.
        logger (logging.Logger): Logger instance.

    Returns:
        None
    """
    logger.info('EPOCH: [%d/%d] BATCHES [%d/%d] loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                (epoch + 1, epochs, i + 1, len_train_loader,
                 (partial_loss1 + partial_loss2) / log_period,
                 partial_loss1 / log_period,
                 partial_loss2 / log_period))


def save_model(model, output_dir, j, logger):
    """
    Saves the model to the specified output directory.

    Args:
        model (torch.nn.Module): The model to be saved.
        output_dir (str): The directory where the model will be saved.
        j (int): Model save counter.
        logger (logging.Logger): Logger instance.

    Returns:
        None
    """
    output_filename = f"{output_dir}/all_80_softmax_{j}_testing_model_type.pt"
    torch.save(model.state_dict(), output_filename)
    logger.info('Model saved as: %s' % output_filename)


def do_train(cfg, model, train_loader, optimizer, losses):
    """
    Perform the training loop for the model.

    Args:
        cfg (object): Configuration object containing training settings.
        model (object): The model to be trained.
        train_loader (object): The data loader for training data.
        optimizer (object): The optimizer used for training.
        losses (list): List of loss functions used for training.

    Returns:
        None
    """
    output_dir = cfg.OUTPUT_DIR
    device = setup_device(cfg)
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD

    logger = logging.getLogger("model.train")
    logger.info("Start training")

    model = model.to(device)
    j = 0

    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        partial_loss1 = 0.0
        partial_loss2 = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, value, explanation = data[0].to(device), data[1].to(device), data[2].to(device)
            out1, out2 = model(inputs)

            explanation = explanation.type(torch.LongTensor).to(device)

            loss1 = losses[0](out1, value)
            loss2 = losses[1](out2, explanation)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            partial_loss1 += loss1.item()
            partial_loss2 += loss2.item()

            if i % log_period == log_period - 1:
                log_training_info(epoch, epochs, i, len(train_loader), partial_loss1, partial_loss2, log_period, logger)
                partial_loss1 = 0.0
                partial_loss2 = 0.0

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        logger.info('EPOCH: [%d] FINISHED loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                    (epoch + 1, train_epoch_loss, running_loss1 / len(train_loader), running_loss2 / len(train_loader)))

        j += 1

        if j % 5 == 0 or j == 1:
            logger.info('Finished Training')
            logger.info('Saving model ...')
            save_model(model, output_dir, j, logger)

    # Generowanie wykresu strat treningowych
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 8))

    plt.plot(epochs_range, train_losses, label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.show()
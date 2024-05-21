# Import necessary modules
import logging
import torch


def do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        losses,
):
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

    # Set the output directory for saving models
    output_dir = cfg.OUTPUT_DIR
   
    # Set device type for training
    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
    # Get training parameters from config
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    
    # Set up logging
    logger = logging.getLogger("model.train")
    logger.info("Start training")
    
    # Move the model to the specified device
    model = model.to(device)
    j = 0  # Counter for saving the model

    # Loop over the dataset for multiple epochs
    for epoch in range(epochs):
        # Initialize running loss variables
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        partial_loss1 = 0.0
        partial_loss2 = 0.0

        # Iterate over the data in the train_loader
        for i, data in enumerate(train_loader, 0):
            # Move inputs, values, and explanations to the device
            inputs, value, explanation = data[0].to(device), data[1].to(device), data[2].to(device)
            
            # Forward pass
            out1, out2 = model(inputs)

            # Convert explanations to LongTensor and move to the device
            explanation = explanation.type(torch.LongTensor).to(device)
            
            # Calculate losses
            loss1 = losses[0](out1, value)
            loss2 = losses[1](out2, explanation)
            loss = loss1 + loss2

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate running losses
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            partial_loss1 += loss1.item()
            partial_loss2 += loss2.item()

            # Log progress periodically
            if i % log_period == log_period - 1:
                logger.info('EPOCH: [%d/%d] BATCHES [%d/%d] loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                        (epoch + 1, epochs, i + 1, len(train_loader), 
                         (partial_loss1 + partial_loss2) / log_period, 
                         partial_loss1 / log_period, 
                         partial_loss2 / log_period))
                partial_loss1 = 0.0
                partial_loss2 = 0.0

        # Log epoch statistics
        logger.info('EPOCH: [%d] FINISHED loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f' %
                    (epoch + 1, running_loss / len(train_loader), 
                     running_loss1 / len(train_loader), 
                     running_loss2 / len(train_loader)))
        
        # Reset running losses
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        j += 1

        # Save the model periodically
        if j % log_period == 0:
            logger.info('Finished Training')
            logger.info('Saving model ...')
            output_filename = output_dir + '/' + "all_80_softmax_" + str(j) + '_testing_model_type.pt'
            torch.save(model.state_dict(), output_filename) 
            logger.info('Model saved as :' + output_filename)

# Import necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.roberta import RobertaModel


class RobertaISTS(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate, hidden_neurons, params_to_optimize=80):
        """
        Initializes the RobertaISTS model.

        Args:
            num_classes (int): The number of classes for classification.
            dropout_rate (float): The dropout rate for regularization.
            hidden_neurons (int): The number of hidden neurons in the linear layers.
            params_to_optimize (int): The number of parameters to optimize in the Roberta model.
        """
        super(RobertaISTS, self).__init__()

        # Load the pre-trained RoBERTa model
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')

        # Freeze all parameters initially
        i = 0
        for param in self.roberta.parameters():
            i += 1
            param.requires_grad = False

        # Unfreeze the last `params_to_optimize` parameters
        j = 0
        for param in self.roberta.parameters():
            j += 1
            if j >= i - params_to_optimize:
                param.requires_grad = True

        # Define the linear layers and dropout layers for the model
        self.linear1 = nn.Linear(in_features=1024, out_features=hidden_neurons)
        self.linear2 = nn.Linear(in_features=hidden_neurons, out_features=1)
        self.linear3 = nn.Linear(in_features=1024, out_features=hidden_neurons)
        self.linear4 = nn.Linear(in_features=hidden_neurons, out_features=num_classes)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        """
        Performs forward pass of the RobertaISTS model.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            tuple: A tuple containing the output for STS value prediction and explanatory layer.
        """
        # Extract features from RoBERTa
        features = self.roberta.extract_features(x)
        # Mean pooling of features
        x = torch.mean(features, 1)
        # Apply dropout
        x = self.dropout1(x)

        # STS value prediction path
        x1 = self.relu1(self.linear1(x))
        x1 = self.dropout2(x1)
        out1 = self.linear2(x1).squeeze(-1)  # Regression output

        # Explanatory layer path
        x2 = self.relu2(self.linear3(x))
        out2 = F.log_softmax(self.linear4(x2), dim=1)  # Classification output

        return out1, out2  # Return both regression and classification outputs

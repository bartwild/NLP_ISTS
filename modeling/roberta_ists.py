import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.roberta import RobertaModel

class RobertaISTS(nn.Module):
    def __init__(self, num_classes, dropout_rate, hidden_neurons):
        super(RobertaISTS, self).__init__()

        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(in_features=1024, out_features=hidden_neurons, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_neurons)  # Batch normalization layer
        self.linear2 = nn.Linear(in_features=hidden_neurons, out_features=1, bias=True)
        
        self.linear3 = nn.Linear(in_features=1024, out_features=hidden_neurons, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_neurons)  # Batch normalization layer
        self.linear4 = nn.Linear(in_features=hidden_neurons, out_features=num_classes, bias=True)
        
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        features = self.roberta.extract_features(x)
        x = torch.mean(features, 1)

        # STS value prediction
        x1 = self.dropout1(x)
        x1 = self.relu1(self.linear1(x1))
        x1 = self.bn1(x1)  # Applying batch normalization
        out1 = self.linear2(x1).squeeze(-1)

        # Explanatory layer
        x2 = self.dropout2(x)
        x2 = self.relu2(self.linear3(x2))
        x2 = self.bn2(x2)  # Applying batch normalization
        out2 = F.log_softmax(self.linear4(x2), dim=1)

        return out1, out2

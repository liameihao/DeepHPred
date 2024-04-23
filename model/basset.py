import numpy as np
import torch
import torch.nn as nn

class Basset(nn.Module):
    def __init__(self, sequence_length=1000, n_targets=196):
        super(Basset, self).__init__()
        self.layer1  = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19),
            nn.BatchNorm1d(num_features=300),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3))

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=200, kernel_size=11),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4))

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4))


        self.n_channels = int((np.floor((np.floor((sequence_length-18)/3)-10)/4)-6)/4)
        self.fc1 = nn.Linear(in_features=self.n_channels*200, out_features=1000)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(in_features=1000, out_features=n_targets)
        self.sig3 = nn.Sigmoid()
        
    def forward(self, inputs):
        output = self.layer1(inputs)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = self.relu4(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu5(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        output = self.sig3(output)

        return output
    
def criterion():
    """
    Specify the appropriate loss function (criterion) for this
    model.

    Returns
    -------
    torch.nn._Loss
    """
    return nn.BCELoss()

def get_optimizer(lr):
    """
    Specify an optimizer and its parameters.

    Returns
    -------
    tuple(torch.optim.Optimizer, dict)
        The optimizer class and the dictionary of kwargs that should
        be passed in to the optimizer constructor.

    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
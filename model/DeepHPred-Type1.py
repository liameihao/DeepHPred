"""
DeepSEA architecture (Zhou & Troyanskaya, 2015).
"""
import numpy as np
import torch
import torch.nn as nn


class DeepHP(nn.Module):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepHP, self).__init__()
        pool_kernel_size = 4
        s_conv_kernel_size = 9
        s_padding = 4
        self.s_conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=s_conv_kernel_size, padding=s_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=s_conv_kernel_size, padding=s_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=s_conv_kernel_size, padding=s_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=s_conv_kernel_size, padding=s_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=s_conv_kernel_size, padding=s_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=s_conv_kernel_size, padding=s_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))
        
        b_conv_kernel_size = 19
        b_padding = 9
        self.b_conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=b_conv_kernel_size, padding=b_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=b_conv_kernel_size, padding=b_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=b_conv_kernel_size, padding=b_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=b_conv_kernel_size, padding=b_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=b_conv_kernel_size, padding=b_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=b_conv_kernel_size, padding=b_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))
        
        self.n_channels = int(np.floor(np.floor(sequence_length/4)/4))
        
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        
        out1 = self.s_conv_net(x)
        out2 = self.b_conv_net(x)
        out = out1 + out2
        
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict

def criterion():
    """
    The criterion the model aims to minimize.
    """
    return nn.BCELoss()

def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.
    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})

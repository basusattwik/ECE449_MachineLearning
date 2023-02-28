#import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
                        nn.BatchNorm2d(num_channels),
                        nn.ReLU()
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
                        nn.BatchNorm2d(num_channels)
                        )
        self.relu = nn.ReLU()


    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        out  = self.conv1(x)
        out  = self.conv2(out)
        out += x # skip connection
        out  = self.relu(out)

        return out


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, num_channels, kernel_size = 3, stride = 2, padding = 1, bias=False),
                        nn.BatchNorm2d(num_channels),
                        nn.ReLU()
                        )
        self.maxpool  = nn.MaxPool2d(kernel_size = 2)
        self.resblock = Block(num_channels)
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.fullconn = nn.Linear(num_channels, num_classes)


    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.resblock(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fullconn(x)
        
        return x


        




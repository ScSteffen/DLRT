# import custom layers
from custom_layers.my_conv import *
from custom_layers.Linear_layer_lr import * 
import torch

class Lenet5(torch.nn.Module):
    def __init__(self,device = 'cpu'):
        super(Lenet5, self).__init__()
        self.device = device
        self.layer = torch.nn.Sequential(
            Conv2d_lr(in_channels = 1, out_channels = 20, kernel_size = 5, stride=1,rank = 20,device = self.device),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            Conv2d_lr(in_channels = 20, out_channels = 50, kernel_size = 5, stride=1,rank = 50,device = self.device),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            torch.nn.Flatten(),
            Linear(800,out_features = 500,rank = 500,device = self.device),  
            torch.nn.ReLU(),
            Linear(500,out_features = 10,device = self.device)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


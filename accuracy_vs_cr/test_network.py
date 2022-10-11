#%%
# import custom layers
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS.my_conv import Conv2d_lr
from optimizer_KLS.Linear_layer_lr_new import Linear
import torch

class Lenet5(torch.nn.Module):
    def __init__(self,device = 'cpu',ranks = [20,50,500,None],fixed = True):
        """  
        initializer for Lenet5.
        NEEDED ATTRIBUTES TO USE dlr_opt:
        self.layer
        NEEDED METHODS TO USE dlr_opt:
        self.forward : standard forward of the NN
        self.update_step : updates the step of all the low rank layers inside the neural net
        self.populate_gradients : method used to populate the gradients inside the neural network in one unique function
        """
        super(Lenet5, self).__init__()
        self.device = device
        self.layer = torch.nn.Sequential(
            Conv2d_lr(in_channels = 1, out_channels = 20, kernel_size = 5, stride=1,rank = ranks[0],device = self.device,fixed = fixed),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            Conv2d_lr(in_channels = 20, out_channels = 50, kernel_size = 5, stride=1,rank = ranks[1],device = self.device,fixed = fixed),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            torch.nn.Flatten(),
            Linear(800,out_features = 500,rank = ranks[2],device = self.device,fixed = fixed),  
            torch.nn.ReLU(),
            Linear(500,out_features = 10,device = self.device)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def update_step(self,new_step = 'K'):
        for l in self.layer:
            if hasattr(l,'lr') and l.lr:
                l.step = new_step

    def populate_gradients(self,x,y,criterion,step = 'all'):

        if step == 'all':
        
            self.update_step(new_step = 'K')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            self.update_step(new_step = 'L')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            return loss,output.detach()

        else:
            
            self.update_step(new_step = step)
            loss = criterion(self.forward(x),y)
            return loss



class Lenet5_cifar10(torch.nn.Module):
    def __init__(self,device = 'cpu',ranks = [20,50,500,None],fixed = True):
        """  
        initializer for Lenet5.
        NEEDED ATTRIBUTES TO USE dlr_opt:
        self.layer
        NEEDED METHODS TO USE dlr_opt:
        self.forward : standard forward of the NN
        self.update_step : updates the step of all the low rank layers inside the neural net
        self.populate_gradients : method used to populate the gradients inside the neural network in one unique function
        """
        super(Lenet5_cifar10, self).__init__()
        self.device = device
        self.layer = torch.nn.Sequential(
            Conv2d_lr(in_channels = 3, out_channels = 20, kernel_size = 5, stride=1,rank = ranks[0],device = self.device,fixed = fixed),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            Conv2d_lr(in_channels = 20, out_channels = 50, kernel_size = 5, stride=1,rank = ranks[1],device = self.device,fixed = fixed),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            torch.nn.Flatten(),
            Linear(1250,out_features = 500,rank = ranks[2],device = self.device,fixed = fixed),  
            torch.nn.ReLU(),
            Linear(500,out_features = 10,device = self.device)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def update_step(self,new_step = 'K'):
        for l in self.layer:
            if hasattr(l,'lr') and l.lr:
                l.step = new_step

    def populate_gradients(self,x,y,criterion,step = 'all'):

        if step == 'all':
        
            self.update_step(new_step = 'K')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            self.update_step(new_step = 'L')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            return loss,output.detach()

        else:
            
            self.update_step(new_step = step)
            loss = criterion(self.forward(x),y)
            return loss

# imports 
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
import warnings
warnings.filterwarnings("ignore", category=Warning)

# low rank convolution class 

class Conv2d_lr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1,bias = True,
            reshaping_type='3',tucker_dlr=False,rank = None,dtype = None,device = None,load_weights = None)->None:
            
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv2d_lr, self).__init__()

        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size,int) else kernel_size
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = dilation if type(dilation)==tuple else (dilation, dilation)
        self.padding = padding if type(padding) == tuple else(padding, padding)
        self.stride = (stride if type(stride)==tuple else (stride, stride))
        self.in_channels = in_channels
        self.reshaping_type = reshaping_type
        self.tucker_dlr = tucker_dlr
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.load_weights = load_weights
        self.weight = torch.nn.Parameter(torch.empty(tuple([self.out_channels, self.in_channels] +self.kernel_size),**factory_kwargs),requires_grad = True)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels,**factory_kwargs))
        else:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels,**factory_kwargs))
    
        # Weights and Bias initialization
        if self.load_weights == None:
            self.reset_parameters()
        else:
            param,b = self.load_weights
            self.bias = torch.nn.Parameter(b)
            self.weight = torch.nn.Parameter(param,requires_grad = True)

        self.original_shape = self.weight.shape
        w =  self.weight.reshape(out_channels,self.in_channels*self.kernel_size_number)
        if rank!=None:
            r = min([self.rank,min(list(w.shape))])
        else:
            r = min(list(w.shape))
        U,S,V = tuple(torch.linalg.svd(w))                         
        U = U[:,0:r].to(device)  
        S = torch.diag(S[0:r]).to(device)                                                           
        V = V.T[:,0:r].to(device)                                                                                                        
        setattr(self.weight,'USV',(U,S,V))
        setattr(self.weight,'tucker_dlr',self.tucker_dlr)           
        setattr(self.weight,'reshaping_type',self.reshaping_type)
        setattr(self.weight,'original_shape',self.original_shape)
        setattr(self.weight,'lr',True)
        setattr(self.weight,'reshape_shape',(self.out_channels,self.in_channels*self.kernel_size_number))
        setattr(self.weight,'rank',self.rank)


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
         # for testing
        # self.original_weight = Parameter(self.weight.reshape(self.original_shape))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)  


    def forward(self, input):

            weight = self.weight
            bias = self.bias
        
            batch_size,_,_,_ = input.shape
            U,S,V = weight.USV
 

            inp_unf = F.unfold(input,self.kernel_size,padding = self.padding,stride = self.stride).to(self.device)
  
            if bias is None:
                out_unf = (inp_unf.transpose(1, 2).matmul(V) )
                out_unf = (out_unf.matmul(S.t()))
                out_unf = (out_unf.matmul(U.t()) + bias).transpose(1, 2)
            else:
                out_h = int(np.floor(((input.shape[2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0])+1))
                out_w = int(np.floor(((input.shape[3]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1])+1))

                out_unf = (inp_unf.transpose(1, 2).matmul(V) )
                out_unf = (out_unf.matmul(S.t()))
                out_unf = (out_unf.matmul(U.t()) + bias).transpose(1, 2)
  
            return out_unf.view(batch_size, self.out_channels, out_h, out_w)

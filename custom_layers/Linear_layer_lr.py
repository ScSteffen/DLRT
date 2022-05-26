# redefinition of the linear layer, adding decomposition attributes
# to the weight object

import math       
import torch
from torch import Tensor
import torch.nn.init as init
from .linear_activation import LinearFunction


class Linear(torch.nn.Module):
    

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,rank = None,load_weights = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.rank = rank
        self.load_weights = load_weights

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        if self.load_weights == None:
            self.reset_parameters()
        else:
            param,b = self.load_weights
            self.weight = torch.nn.Parameter(param)
            if bias:
                self.bias = torch.nn.Parameter(b)
            else:
                self.register_parameter('bias', None)

        if self.rank == None:
            r = min([in_features,out_features])
            U,S,V = tuple(torch.linalg.svd(self.weight))
            U = torch.nn.Parameter(U[:,0:r],requires_grad = False)
            S = torch.nn.Parameter(torch.diag(S[0:r]),requires_grad = False)
            V = V.T[:,0:r]
            V = torch.nn.Parameter(V,requires_grad = False)
            setattr(self.weight,'USV',(U,S,V))   # adding attributes to the weight
            
        elif type(self.rank) == int and self.rank<=torch.min(torch.tensor(self.weight.shape)):
            U,S,V = tuple(torch.linalg.svd(self.weight))
            U = torch.nn.Parameter(U[:,0:self.rank],requires_grad = False)
            S = torch.nn.Parameter(torch.diag(S[0:self.rank]),requires_grad = False)
            V = V.T[:,0:self.rank]
            V = torch.nn.Parameter(V,requires_grad = False)
            setattr(self.weight,'USV',(U,S,V))   # adding attributes to the weight
        
        setattr(self.weight,'lr',True)
        setattr(self.weight,'rank',self.rank)



    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        x = LinearFunction.apply(input,self.weight,self.bias)
        return x 

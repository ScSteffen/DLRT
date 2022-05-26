# redefine the Linear activation function in order to use the factors of the decomposition
# This allows a gain in computational efficiency cause the multiplication cost is overall smaller
# if the rank is small enough

import torch
# Inherit from Function
class LinearFunction(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        #print(f'input device {input.device}')
        if hasattr(weight,'USV'):
            U,S,V = weight.USV
            #print(f'self U device linear {weight.USV[0].device}')
            #print(f' U device linear {U.device}')
            output = input.mm(V)
            output = output.mm(S.T)   
            output = output.mm(U.T)
        else:
            output = input.mm(weight.T)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if hasattr(weight,'USV'):
                U,S,V = weight.USV
                grad_input = grad_output.mm(U)
                grad_input = grad_input.mm(S)
                grad_input = grad_input.mm(V.T) 
            else:
                grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias
import torch
from torch import Tensor
from typing import List, Optional   
import torch.optim as optim
from copy import deepcopy
from .low_rank_integrators import * 

# construction of the optimizer class

class dlr_optim(optim.Optimizer):  
    

    def __init__(self, NN, lr=1e-1, momentum=0, dampening=0,loss_function = None,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,name = 'ksl',theta = 0.1,integrator = 'euler',absolute_theta = True):    # put the batch just in the step method
        if lr < 0.0:
             raise ValueError("Invalid learning rate: {}".format(lr))

        params = list(NN.parameters()) 
        self.low_rank_train_bool = [True if (hasattr(p,'lr')) else False for p in NN.parameters()]
        self.name = name
        self.NN = NN
        self.lr = lr
        self.theta = theta
        self.momentum = momentum
        self.maximize = maximize
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.integrator = integrator
        self.loss_function  = loss_function
        self.absolute_theta = absolute_theta
        self.tucker_lr = False

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(NN = NN,lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,name = name,theta = theta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.history = dict()

        if integrator.lower() == 'adams_bashforth':
          self.history['last_weight'] = None
          self.history['last_grad'] = None
        
        super(dlr_optim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, batch = None ,closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.dlr(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'],
                batch = batch)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


    def dlr(self,params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            has_sparse_grad: bool = None,
            foreach: bool = None,
            batch = None):

        if foreach is None:
            # Placeholder for more complex foreach logic to be added when value is not set
            foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        if foreach and not torch.jit.is_scripting():
            func = self._multi_tensor_dlr
        else:
            func = self._single_tensor_dlr

        func(params,
             d_p_list,
             momentum_buffer_list,
             has_sparse_grad=has_sparse_grad,
             batch = batch )

    def _single_tensor_dlr(self,params: List[Tensor],
                           d_p_list: List[Tensor],
                           momentum_buffer_list: List[Optional[Tensor]],
                           *,
                           has_sparse_grad: bool,
                           batch : tuple):


        new_svd = []

        x,y = batch


        if self.integrator == 'adams_bashforth' and self.history['last_weight'] == None:

          self.history['last_weight'] = [(p,p.USV) if hasattr(p,'USV') else (p,None) for p in self.NN.parameters() ]  # changed
          self.history['last_grad'] = d_p_list
          euler_step_integrator = dlr_optim(NN = self.NN, lr = self.lr, momentum = self.momentum, dampening = self.dampening, loss_function  = self.loss_function,
                 weight_decay = self.weight_decay, nesterov = self.nesterov, maximize = self.maximize,name = self.name,theta = self.theta,integrator = 'euler')
          self.zero_grad()
          with torch.set_grad_enabled(True):
            current_loss = self.loss_function(self.NN(x),y)
          current_loss.backward()
          euler_step_integrator.step(batch = (x,y))
          d_p_list = [p.grad.data for p in self.NN.parameters()]
          params = [p for p in self.NN.parameters()]

        

        alpha = self.lr if self.maximize else -self.lr

        
        for i, param in enumerate(params,0):


            d_p = d_p_list[i]

            if self.low_rank_train_bool[i]:    # if weight is not a bias use the rank constrained optimization
              

              if self.weight_decay != 0: 
                  d_p.add(param, alpha=self.weight_decay)

              if self.momentum != 0:
                  buf = momentum_buffer_list[i]

                  if buf is None:
                      buf = torch.clone(d_p).detach()
                      momentum_buffer_list[i] = buf
                  else:
                      buf.mul_(self.momentum).add_(d_p, alpha=1 - self.dampening)

                  if self.nesterov:
                      d_p = d_p.add(buf, alpha=self.momentum)
                  else:
                      d_p = buf

              if (len(param.shape)<=2 and not hasattr(param,'reshaping_type')) or hasattr(param,'reshaping_type'):

                flag = hasattr(param,'reshaping_type')
                if hasattr(param,'reshaping_type'):
                  d_p = d_p.reshape(param.reshape_shape)

                if self.integrator.lower() == 'euler':    # EULER 

                  if self.name == 'ksl':
                        r = min(param.shape) if param.rank == None else param.rank
                        ksl_out = ksl(alpha,d_p,r,X = None,USV = param.USV)
                        param.USV = ksl_out[1] 

                  elif self.name == 'kls':
                        r = min(param.shape) if param.rank == None else param.rank
                        kls_out_UV = kls_UV_step(alpha,d_p,r, X = None,USV = param.USV)
                        param.USV =  kls_out_UV[1]
                        self.zero_grad()
                        with torch.set_grad_enabled(True):
                          current_loss = self.loss_function(self.NN(x),y)
                        current_loss.backward(retain_graph = True)
                        updated_d_p = list(self.NN.parameters())[i].grad if not flag else list(self.NN.parameters())[i].grad.reshape(param.reshape_shape)
                        kls_out_S = kls_S_step(alpha,kls_out_UV[1],kls_out_UV[2],updated_d_p,param.USV)
                        param.USV = kls_out_S[1] 

                  elif self.name == 'kls_adaptive':
                    if param.rank !=None:
                        kls_out_UV = kls_adaptive_UV_step(alpha,d_p,X = None,USV = param.USV)
                        param.USV = kls_out_UV[3] 
                        self.zero_grad()
                        with torch.set_grad_enabled(True):
                          current_loss = self.loss_function(self.NN(x),y)
                        current_loss.backward(retain_graph = True) 
                        current_theta = self.theta if (len(param.shape)>2) else 0.45*self.theta
                        updated_d_p = list(self.NN.parameters())[i].grad if not flag else list(self.NN.parameters())[i].grad.reshape(param.reshape_shape)
                        kls_out_S = kls_adaptive_S_step(alpha,kls_out_UV[1],kls_out_UV[2],updated_d_p,current_theta,absolute=self.absolute_theta)
                        param.USV = kls_out_S[1] 
                    else:  
                        r = min(d_p.shape)
                        kls_out_UV = kls_UV_step(alpha,d_p,r, X = None,USV = param.USV)
                        param.USV = kls_out_UV[1] 
                        self.zero_grad()
                        with torch.set_grad_enabled(True):
                          current_loss = self.loss_function(self.NN(x),y)
                        current_loss.backward(retain_graph = True)
                        updated_d_p = list(self.NN.parameters())[i].grad if not flag else list(self.NN.parameters())[i].grad.reshape(param.reshape_shape)
                        kls_out_S = kls_S_step(alpha,kls_out_UV[1],kls_out_UV[2],updated_d_p,param.USV)
                        param.USV = kls_out_S[1] 

                  else : 
                        raise ValueError('invalide optimizer name {}:'.format(self.name)) 



                elif self.integrator.lower() == 'adams_bashforth':   # Adams-bashforth (DISCRETIZED DIRECTLY IN THE THETA SPACE)
                  if self.name == 'ksl':
                        updated_d_p = -( -(3/2)*d_p+(1/2)*self.history['last_grad'][i] )
                        ksl_out = ksl(alpha,updated_d_p,self.rank_list[i],X = None,USV = param.USV)
                        param.USV = ksl_out[1]
                        self.history['last_weight'][i] = param
                        self.history['last_grad'][i] = d_p
                  elif self.name == 'kls':
                        to_save_weight = param
                        to_save_grad = d_p
                        updated_d_p = -( -(3/2)*d_p+(1/2)*self.history['last_grad'][i] )
                        kls_out_UV = kls_UV_step(alpha,updated_d_p,self.rank_list[i], X = None,USV = param.USV)
                        param.USV = kls_out_UV[1]
                        self.zero_grad()
                        with torch.set_grad_enabled(True):
                          current_loss = self.loss_function(self.NN(x),y)
                        current_loss.backward()
                        updated_d_p = list(self.NN.parameters())[i].grad
                        kls_out_S = kls_S_step(alpha,kls_out_UV[1],kls_out_UV[2],updated_d_p,self.rank_list[i])
                        param.USV = kls_out_S[1]
                        self.history['last_weight'][i] = param
                        self.history['last_grad'][i] = d_p
                  elif self.name == 'kls_adaptive':
                    if param.rank != None:
                        to_save_weight = deepcopy(param)
                        to_save_grad = deepcopy(d_p)
                        updated_d_p = -( -(3/2)*d_p+(1/2)*self.history['last_grad'][i])
                        kls_out_UV = kls_adaptive_UV_step(alpha,updated_d_p,X = None,USV = param.USV)
                        param.USV = kls_out_UV[3]
                        self.zero_grad()
                        with torch.set_grad_enabled(True):
                          current_loss = self.loss_function(self.NN(x),y)
                        current_loss.backward()
                        current_theta = self.theta if (len(param.shape)>2) else 0.45*self.theta
                        updated_d_p = list(self.NN.parameters())[i].grad.data
                        kls_out_S = kls_adaptive_S_step(alpha,kls_out_UV[1],kls_out_UV[2],updated_d_p,current_theta,absolute=self.absolute_theta)
                        param.USV  = kls_out_S[1]
                        self.history['last_weight'][i] = to_save_weight
                        self.history['last_grad'][i] = to_save_grad
                    else:
                        r = min(d_p.shape)
                        to_save_weight = deepcopy(param)
                        to_save_grad = deepcopy(d_p)
                        updated_d_p = -( -(3/2)*d_p+(1/2)*self.history['last_grad'][i])
                        kls_out_UV =  kls_UV_step(alpha,updated_d_p,r, X = None,USV = param.USV)
                        param.USV = kls_out_UV[1] 
                        self.zero_grad()
                        with torch.set_grad_enabled(True):
                          current_loss = self.loss_function(self.NN(x),y)
                        current_loss.backward()
                        updated_d_p = list(self.NN.parameters())[i].grad if not flag else list(self.NN.parameters())[i].grad.reshape(param.reshape_shape)
                        kls_out_S = kls_S_step(alpha,kls_out_UV[1],kls_out_UV[2],updated_d_p,param.USV)
                        param.USV = kls_out_S[1] 
                        self.history['last_weight'][i] = to_save_weight
                        self.history['last_grad'][i] = to_save_grad
                               
                        
                        
                  else : 
                        raise ValueError('invalide optimizer name {}:'.format(self.name))  

                else:
                  raise ValueError('invalid integrator name {}'.format(self.integrator))



            else:   # else for non low rank parameters, use unconstrained optimization

            
              if self.weight_decay != 0: 
              
                  d_p.add(param, alpha=self.weight_decay)

              if self.momentum != 0:
                  buf = momentum_buffer_list[i]

                  if buf is None:
                      buf = torch.clone(d_p).detach()
                      momentum_buffer_list[i] = buf
                  else:
                      buf.mul_(self.momentum).add_(d_p, alpha=1 - self.dampening)

                  if self.nesterov:
                      d_p = d_p.add(buf, alpha=self.momentum)
                  else:
                      d_p = buf

              alpha = self.lr if self.maximize else -self.lr
              param.add_(d_p, alpha=alpha)
        torch.cuda.empty_cache()
            


            

# low rank integrators

import torch
import numpy as np

# KSL

def ksl(tau,grad,rank,X = None,USV = None,reconstruct = False):

    """ 
    INPUTS :
    tau : integration step 
    grad : gradient of the loss with respect to a weight 
    rank : rank constraint
    X : weight (optional) 
    USV : svd like decomposition for the weight
    reconstruct : flag if to reconstruct the weight at the end or not
    OUTPUT:
    Y : reconstructed weight after the integration step if reconstruct is True, else none
    (U,S,V) : new decomposition after the KSL integration step
     """


    if USV == None:
      U,S,V = torch.linalg.svd(X)
      U = U[:,0:rank]
      S = S[0:rank]
      V = V.T[:,0:rank]
      S = torch.diag(S)
    else: 
      U,S,V = USV

    
    K = U@S 
    K = K + tau*grad@V
    U,S = torch.linalg.qr(K) 
    
    S = S - tau*U.T@grad@V

    L = V@S.T 
    L = L + tau*grad.T@U
    V,S = torch.linalg.qr(L)
    S = S.T

    if not reconstruct:
      Y = None
    else:
      Y = U@S@V.T


    return Y,(U,S,V)



# KLS

def kls_UV_step(tau,grad,rank, X = None,USV = None,reconstruct = False):

    """ 
    INPUTS :
    tau : integration step 
    grad : gradient of the loss with respect to a weight 
    rank : rank constraint
    X : weight (optional) 
    USV : svd like decomposition for the weight
    reconstruct : flag if to reconstruct the weight at the end or not
    OUTPUT:
    Y : reconstructed weight after the integration step if reconstruct is True, else none
    (U,S,V) : new decomposition after the KL integration step
    (M,N): change of coordinate matrices for S in the S integration step
     """


    if USV == None:
      U_0,S_0,V_0 = torch.linalg.svd(X)
      U_0 = U_0[:,0:rank]
      S_0 = S_0[0:rank]
      V_0 = V_0.T[:,0:rank]
      S_0 = torch.diag(S_0)
    else: 
      U_0,S_0,V_0 = USV

    
    K_0 = U_0@S_0
    L_0 = V_0@S_0.T


    # update K

    K_1 = K_0 + tau*grad@V_0
    U_1,R_1 = torch.linalg.qr(K_1)
    M = U_1.T@U_0

    # update L

    L_1 = L_0+tau*grad.T@U_0
    V_1,R_1 = torch.linalg.qr(L_1)
    N = V_1.T@V_0

    if not reconstruct:
      Y = None
    else:
      Y = U_1@S_0@V_1.T

    return Y,(U_1,S_0,V_1),(M,N)


def kls_S_step(tau,USV,MN,updated_grad,reconstruct=False,UV_fixed = False):

    """ 
    INPUTS :
    tau : integration step 
    updated_grad : gradient of the loss with respect to a weight after the KL update
    USV : svd like decomposition for the weight
    reconstruct : flag if to reconstruct the weight at the end or not
    UV_fixed : flag if U and V are kept fixed to learn just S
    OUTPUT:
    Y : reconstructed weight after the integration step if reconstruct is True, else none
    (U_1,S_1,V_1) : new decomposition after the KLS integration step
     """

    U_1,S_0,V_1 = USV
    M,N = MN

    if not UV_fixed: 

      S_1 = M@S_0@N.T+tau*U_1.T@updated_grad@V_1
    
    else:

      S_1 = S_0+tau*U_1.T@updated_grad@V_1

    if not reconstruct:
      Y = None
    else:
      Y = U_1@S_1@V_1.T

    return Y,(U_1,S_1,V_1)



# KLS_ADAPTIVE

def kls_adaptive_UV_step(tau,grad,X = None,USV = None,reconstruct = False):

    """ 
    INPUTS :
    tau : integration step 
    grad : gradient of the loss with respect to a weight 
    X : weight (optional) 
    USV : svd like decomposition for the weight
    reconstruct : flag if to reconstruct the weight at the end or not
    OUTPUT:
    Y : reconstructed weight after the integration step if reconstruct is True, else none
    (U_hat,S_0,V_hat) : New svd like decomposition after the adaptive KL integration step
    (M_hat,N_hat) : change of coordinate matrices for the S update
    (U_hat@M_hat,S_0,V_hat@N_hat) : transformed svd like decomposition
     """



    if USV == None:
      U_0,S_0,V_0 = torch.linalg.svd(X)
      n,m = U_0.shape[0],V_0.shape[1]
      rank = torch.min([n,m])
      U_0 = U_0[:,0:rank]
      S_0 = S_0[0:rank]
      V_0 = V_0.T[:,0:rank]
      S_0 = torch.diag(S_0)
    else: 
      U_0,S_0,V_0 = USV

    
    K_0 = U_0@S_0
    L_0 = V_0@S_0.T


    # update K

    K_1 = K_0 + tau*grad@V_0
    U_hat = torch.hstack((K_1,U_0))
    try:
      U_hat,_ = torch.linalg.qr(U_hat)
    except:
      U_hat,_ = np.linalg.qr(U_hat)
      U_hat = torch.tensor(U_hat)
    M_hat = U_hat.T@U_0

    # update L

    L_1 = L_0+tau*grad.T@U_0
    V_hat = torch.hstack((L_1,V_0))
    try :
      V_hat,_ = torch.linalg.qr(V_hat)
    except:
      V_hat,_ = np.linalg.qr(V_hat.detach().numpy())
      V_hat= torch.tensor(V_hat)
    N_hat = V_hat.T@V_0

    
    if not reconstruct:
      Y = None 
    else:
      Y = U_hat@M_hat@S_0@N_hat.T@V_hat.T


    return Y,(U_hat,S_0,V_hat),(M_hat,N_hat),(U_hat@M_hat,S_0,V_hat@N_hat)


def kls_adaptive_S_step(tau,USV,MN,updated_grad,theta,absolute = True,reconstruct = False):

    """ 
    INPUTS :
    tau : integration step 
    updated_grad : gradient of the loss with respect to a weight after KL integration step 
    MN: change of coordinate matrices for the S update
    USV : svd like decomposition for the weight
    theta : threshold for the unconventional integrator
    absolute : flag if to use absolute or relative threshold theta
    reconstruct : flag if to reconstruct the weight at the end or not
    OUTPUT:
    Y : reconstructed weight after the integration step if reconstruct is True, else none
    (U_1,S_1,V_1) : new decomposition after the KLS adaptive integration step
     """

    U_hat,S_0,V_hat = USV
    M_hat,N_hat = MN

    S_hat_1 = M_hat@S_0@N_hat.T+tau*U_hat.T@updated_grad@V_hat

    try:
      P_hat,sigma_hat,Q_hat = torch.linalg.svd(S_hat_1)
    except Exception as e:
      print(e)
      P_hat,sigma_hat,Q_hat = np.linalg.svd(S_hat_1.detach().numpy())
      P_hat,sigma_hat,Q_hat = torch.tensor(P_hat),torch.tensor(sigma_hat),torch.tensor(Q_hat)

    threshold_index = 0

    if not absolute:
      
      scaled_theta = theta*torch.linalg.norm(sigma_hat)

    else:

      scaled_theta = theta

    for i_threshold in range(sigma_hat.shape[0]):

      relative_residual_norm = torch.linalg.norm(sigma_hat[i_threshold+1::])

      if relative_residual_norm<=scaled_theta:

        threshold_index = i_threshold+1
        break

    P_1 = P_hat[:,0:threshold_index]
    Q_1 = Q_hat[:,0:threshold_index]
    S_1 = torch.diag(sigma_hat[0:threshold_index])
    U_1 = U_hat@P_1
    V_1 = V_hat@Q_1

    if not reconstruct:
      Y = None
    else:
      Y = U_1@S_1@V_1.T

    return Y,(U_1,S_1,V_1)

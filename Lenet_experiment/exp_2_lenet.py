import pandas as pd
import torch
import numpy as np
import os 
import sys 

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import optimizer.dlr_optimizer as dlr_optimizer
from train_save_exp2 import train_save
import tensorflow as tf
from Lenet5 import Lenet5
from sklearn.model_selection import train_test_split
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


path = 'results_Lenet5/'

MAX_EPOCHS = 120

def accuracy(y_hat,y):
  return torch.mean((y_hat == y).to(torch.float32))



thetas = [0.11,0.15,0.2,0.3]


metric  = accuracy
criterion = torch.nn.CrossEntropyLoss(reduction='mean') 
metric_name = 'accuracy'



for index,theta in enumerate(thetas,0):


  for cv_run in range(5):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    x = np.vstack([x_train,x_test])
    y = np.hstack([y_train,y_test])
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=60000,stratify = y)
    
    ## for cifar
    x_train,x_test = x_train.reshape(x_train.shape[0],1,x_train.shape[1],x_train.shape[2]),x_test.reshape(x_test.shape[0],1,x_test.shape[1],x_test.shape[2])  
    y_train,y_test = y_train.reshape(y_train.shape[0]),y_test.reshape(y_test.shape[0])
    ##
    
    x_train,x_test,y_train,y_test = torch.tensor(x_train).float()/255,torch.tensor(x_test).float()/255,torch.tensor(y_train),torch.tensor(y_test)


    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 50000,stratify = y_train)
    
    ##
    print(f'train shape {x_train.shape}')
    print(f'val shape {x_val.shape}')
    print(f'test shape {x_test.shape}')
    
    
    batch_size_train,batch_size_test = 64,64
    
    train_loader = torch.utils.data.DataLoader(
      [(x_train[i],y_train[i]) for i in range(x_train.shape[0])],
      batch_size=batch_size_train, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
      [(x_val[i],y_val[i]) for i in range(x_val.shape[0])],
      batch_size=batch_size_test, shuffle=True)
    
    final_test_loader = torch.utils.data.DataLoader(
    [(x_test[i],y_test[i]) for i in range(x_test.shape[0])],
    batch_size=batch_size_test, shuffle=True)


    f = Lenet5(device = device)
    f = f.to(device)
    optimizer = dlr_optimizer.dlr_optim(f,lr = 0.05,name = 'kls_adaptive',theta = theta,loss_function = criterion,integrator = 'euler',absolute_theta = False)
    run = optimizer.name
    if optimizer.name == 'kls_adaptive':
      run+='_'+str(index)+str(optimizer.theta)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    print('='*100)
    print(f'run number {index} \n theta = {theta}')
    try:
      train_results = train_save(f,train_loader,test_loader,final_test_loader,criterion,optimizer,scheduler = scheduler,\
                          epochs = MAX_EPOCHS,metric = metric ,metric_name = metric_name,path = path,device = device,cv = str(cv_run))
    except Exception as e:
      print(e)
      print('training went bad')





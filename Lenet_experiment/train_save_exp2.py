import os 
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch



def train_save(NN,trainloader,testloader,finaltestloader,criterion,optimizer,\
               epochs = 300,metric = None,metric_name = None,dlr = True,scheduler = None,path = None,device = 'cpu',cv = '0'):  

  '''
  INPUTS: 
  NN : Pytorch neural network object 
  trainloader : trainloader generator for the data
  testloader : validation data generator
  finaltestloader : test data generator
  criterion : loss function
  optimizer : optimizer for the training procedure
  epochs : number of training epochs
  metric : evalutation metric 
  metric_name : name of the metric for the prints
  dlr : flag variable, True if the integrator is a low rank one otherwise False
  scheduler : scheduler for the adaptive learning rate
  path : path to where the results will be saved
  device : device variable to decide if to train using cpu or cuda
  cv : index for the different inizializations to save them with different names
  OUTPUT: 
  history : dictionary containing train and validation loss/metric.
  '''
  running_data = pd.DataFrame(data = None,columns = ['epoch','theta','train_loss','train_'+metric_name,'validation_loss','validation_'+metric_name,'test_'+metric_name,\
                                                     'ranks','# effective parameters','cr_test','# effective parameters train','cr_train'])
  total_params = sum([int(torch.prod(torch.tensor(p.shape))) for p in NN.parameters() if hasattr(p,'USV')])
  loss_list = []
  loss_list_test = []

  best_loss = None # not saving best weight
  best_weight = None

  run = optimizer.name
  if optimizer.name == 'kls_adaptive':
    run+='_'+str(optimizer.theta)+'_'+cv
  
  print('start training...\n')
  for epoch in tqdm(range(epochs)):
    
    running_loss = 0.0
    running_metric = 0.0
    k =  len(trainloader)
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs,labels = inputs.to(device),labels.to(device)
      # zero the parameter gradient
            
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs = NN(inputs).to(device)
      loss = criterion(outputs, labels) 
      running_loss += loss/k

      if metric!=None:
        running_metric += metric(torch.argmax(outputs.detach(),axis = 1),labels.detach().to(torch.int32))/k
      loss.backward(retain_graph = True)
      if dlr :
        optimizer.step(batch = (inputs,labels))
        if scheduler!=None:
          scheduler.step()
      else:
        optimizer.step()
        if scheduler!=None:
          scheduler.step()
      
      del data
      
      if ((epoch%5 == 0 and i == k-1) or ( epoch == epochs-1 and i == k-1)) and dlr:
        if path !=None:
          running_data.to_csv(path+'/running_data_'+str(run)+'.csv')
        #for i,p in enumerate(NN.named_parameters()):
        #  n,p = p
         # if 'bias' not in n and hasattr(p,'USV') and len(p.shape)==2:
         #   list(NN.parameters())[i] = p.USV[0]@p.USV[1]@(p.USV[2].T)
         #   list(NN.parameters())[i].data = p.USV[0]@p.USV[1]@(p.USV[2].T)
         #   p = p.USV[0]@p.USV[1]@(p.USV[2].T)
         # if 'bias' not in n and hasattr(p,'USV') and len(p.shape)>2:
         #   w = (p.USV[0]@p.USV[1]@(p.USV[2].T)).reshape(p.shape)
          #  list(NN.parameters())[i] = w
          #  list(NN.parameters())[i].data = w
            
        if path !=None:
          #torch.save(NN.state_dict(),path+'/model_'+str(run)+'.pth')
          pass
        
        print('-'*100)
        print(f'layer ranks at epoch {epoch}:\n')
        for j,p in [(el[0],el[1][1]) for el in enumerate(NN.named_parameters()) if (('bias' not in el[1][0]) and 'bn' not in el[1][0])]:
          if len(p.shape)==2:
              print(f'layer {j//2 + 1} rank: {torch.linalg.matrix_rank(p.USV[1])}')
          elif len(p.shape)>2 and optimizer.tucker_lr == True:
              print(f'layer {j//2 + 1} rank: {p.tucker_decomposition[0].shape}')
          if len(p.shape)>2 and hasattr(p,'USV'):
              print(f'layer {j//2 + 1} rank: {torch.linalg.matrix_rank(p.USV[1])}')

          

        print('-'*100)
       


    with torch.no_grad():
      running_test_loss = 0.0
      running_test_metric = 0.0
      k =  len(testloader)
      for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = NN(inputs)
        running_test_loss += criterion(outputs, labels)/k
        if metric!=None:
          running_test_metric += metric(torch.argmax(outputs.detach(),axis = 1),labels.detach().to(torch.int32))/k


    with torch.no_grad():
      running_finaltest_metric = 0.0
      k =  len(finaltestloader)
      for i, data in enumerate(finaltestloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = NN(inputs)
        if metric!=None:
          running_finaltest_metric += metric(torch.argmax(outputs.detach(),axis = 1),labels.detach().to(torch.int32))/k
          
        


    loss_list.append(running_loss.detach().cpu().numpy())
    loss_list_test.append(running_test_loss.detach().cpu().numpy())

    if epoch == 0:

      best_loss = running_test_loss

    else:

      if best_loss>running_test_loss:

        best_loss = running_test_loss

        #for i,p in enumerate(NN.named_parameters()):
            #n,p = p
            #if 'bias' not in n and hasattr(p,'USV'):
              #w = (p.USV[0]@p.USV[1]@(p.USV[2].T)).reshape(p.shape)
              #list(NN.parameters())[i] = w
              #list(NN.parameters())[i].data = w
              #p.data = w
        if path !=None:
          #torch.save(NN.state_dict(),path+'/bestmodel_'+str(run)+'.pth')
          pass


    if metric==None:

      print('epoch ['+str(epoch)+']'+'loss: '+str(torch.round(running_loss,4))+'  test loss: '+ str(torch.round(running_test_loss,4)))
      print('='*100)

    else:

      print(f'epoch[{epoch}]: loss: {running_loss:9.4f} | {metric_name}: {running_metric:9.4f} | test loss: {running_test_loss:9.4f} | test {metric_name}:{running_test_metric:9.4f}')
      print('='*100)
      
    ranks = [int(torch.linalg.matrix_rank(p.USV[1])) for n,p in NN.named_parameters() if hasattr(p,'USV')]
    test_params = sum([sum([(p.USV[0].shape[0]+p.USV[2].shape[0])*p.USV[1].shape[0]]) for n,p in NN.named_parameters() if hasattr(p,'USV')])
    train_params = sum([sum([(p.USV[0].shape[0]+p.USV[2].shape[0]+p.USV[1].shape[0])*p.USV[1].shape[0]]) for n,p in NN.named_parameters() if hasattr(p,'USV')])
    cr_test = test_params/total_params
    cr_train = train_params/total_params
    if dlr:
      running_data.loc[epoch] = [epoch,optimizer.theta,float(running_loss.cpu()),float(running_metric.cpu()),float(running_test_loss.cpu()),\
                              float(running_test_metric.cpu()),float(running_finaltest_metric.cpu()),ranks,\
                               test_params,cr_test,train_params,cr_train]

  plt.plot(loss_list,label = 'train')
  plt.plot(loss_list_test,label = 'test')
  plt.title('loss')
  plt.xlabel('epoch')
  plt.ylabel(str(criterion))
  plt.legend(['train','test'])
  plt.show()

  print('\n')
  print('='*80)
  print('\n')
  print('final train loss:',loss_list[len(loss_list)-1])
  print('final test loss:',loss_list_test[len(loss_list_test)-1])
  print(f'final train {metric_name}: {running_metric}')
  print(f'final test {metric_name}: {running_test_metric}')

  history = [float(running_loss.cpu()),float(running_test_loss.cpu())]
  history_keys = ['train_loss','test_loss']

  if metric !=None:

    history += [float(running_metric.cpu()),float(running_test_metric.cpu())]
    history_keys += ['train_'+metric_name,'test_'+metric_name]

  history = dict(zip(history_keys,history))

  print('\nDone!')

  return history

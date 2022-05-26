import pandas as pd
import numpy as np
from os import walk


def extract_best_val(run,metric = 'acc'):    
  if metric == 'acc':
    i = np.argmax(run['validation_accuracy'])
  else:
    i = np.argmin(run['validation_loss'])
  return run.loc[i].to_frame()


results = []         # import files from the folder of the results
path = '../pytorch_dlr/results_Lenet5/'
file_names = list()
for root, dirc, files in walk(path):
    for FileName in files:
      if 'running' in FileName and 'old.csv' not in FileName:
        el = pd.read_csv(path+FileName)
        el.drop(columns = ['Unnamed: 0'],inplace = True)
        results.append(el)
        file_names.append(FileName)
        
        
        
# concatenate the final values in a dataframe
cv_results = []
for i,el in enumerate(results):
  cv_results.append(extract_best_val(el,'loss'))
cv_results = pd.concat(cv_results,axis = 1).transpose()
cv_results.sort_values(by = 'theta',inplace = True)
main_results = pd.read_csv(path+'/results_section5.3.csv')
main_results.drop(index = [4,5],inplace = True)
main_results['theta'] = [float(el) for el in main_results['theta']]
cv_all_results = pd.concat([cv_results,main_results],axis = 0)
cv_all_results.drop(columns = ['Unnamed: 0','epoch','train_loss','train_accuracy'],inplace = True)
cv_all_results.sort_values(by = 'theta',inplace = True)



# create table with the mean values
cross_val_tab = pd.DataFrame(data = None,columns = ['theta','test accuracy mean','test accuracy std','ranks',\
                                                    'params test','cr test',\
                                                    'params train','cr train'
                                                    ])


for i,theta in enumerate([0.11,0.15,0.2,0.3]):

  cross_val_tab.loc[i] = [theta,
                          np.round(cv_all_results.loc[cv_all_results['theta']==theta]['test_accuracy'].mean()*100,3),
                          np.round(cv_all_results.loc[cv_all_results['theta']==theta]['test_accuracy'].std()*100,3),
                          list(cv_all_results.loc[(cv_all_results['theta']==theta) & (cv_all_results['test_accuracy']==\
                                              max(cv_all_results[cv_all_results['theta']==theta]['test_accuracy']))]['ranks'])[0],

                          int(cv_all_results.loc[(cv_all_results['theta']==theta) & (cv_all_results['test_accuracy']==\
                                              max(cv_all_results[cv_all_results['theta']==theta]['test_accuracy']))]['# effective parameters']),

                          np.round(float(cv_all_results.loc[(cv_all_results['theta']==theta) & (cv_all_results['test_accuracy']==\
                                              max(cv_all_results[cv_all_results['theta']==theta]['test_accuracy']))]['cr_test']),3),
                          
                          
                          int(cv_all_results.loc[(cv_all_results['theta']==theta) & (cv_all_results['test_accuracy']==\
                                              max(cv_all_results[cv_all_results['theta']==theta]['test_accuracy']))]['# effective parameters train']),
                          
                          np.round(float(cv_all_results.loc[(cv_all_results['theta']==theta) & (cv_all_results['test_accuracy']==\
                                              max(cv_all_results[cv_all_results['theta']==theta]['test_accuracy']))]['cr_train']),3),
                          ]
  
# save table
pd.DataFrame.to_csv(cross_val_tab,'results_Lenet5/cross_val_tab_paper.csv')
        
        

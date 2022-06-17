#%%
"""
Spyder Editor

This is a temporary script file.
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.pyplot import figure

figure(figsize=(15, 15), dpi=80)
#%% BARPLOTS
for model in ['XGB', 'RF']:
    for index in ['FI', 'SI']:
        for value in ['temp', 'stress']:
            counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\' + model + '_' + index + '_' + value + '.csv')
            counter.columns = ['parameters', 'count']
            
            sns.barplot(data = counter, y = 'parameters', x = 'count', color = 'green').set_title(model + '_' + index + '_' + value)
            plt.show()
            print(model, index, value)
#%% XGB SI TEMP
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\XGBoost_results\\sobol_indicess_temp_point'+str(observe_point)+'.csv',
                           index_col = 0)
    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\XGB_SI_temp.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('XGB_SI_temp_point_' + str(k))
    plt.show()
#%% XGB SI STRESS
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\XGBoost_results\\sobol_indices_stress_point'+str(observe_point)+'.csv',
                           index_col = 0)
    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\XGB_SI_stress.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('XGB_SI_stress_point_' + str(k))
    plt.show()
#%% XGB FI TEMP
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\XGBoost_results\\feature_importances_temp_point'+str(observe_point)+'.csv',
                           index_col = 0)
    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\XGB_FI_temp.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('XGB_FI_temp_point_' + str(k))
    plt.show()
#%% XGB FI STRESS
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\XGBoost_results\\feature_importances_stress_point'+str(observe_point)+'.csv',
                           index_col = 0)
    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\XGB_FI_stress.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('XGB_FI_stress_point_' + str(k))
    plt.show()
#%% RANDOM FOREST SI TEMP
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\RandomForest_results\\sobol_indicess_temp_point'+str(observe_point)+'.csv',
                           index_col = 0)
    
    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\RF_SI_temp.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('RandomForest_SI_temp_point_' + str(k))
    plt.show()
#%% RANDOM FOREST SI STRESS
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\RandomForest_results\\sobol_indices_stress_point'+str(observe_point)+'.csv',
                           index_col = 0)

    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\RF_SI_stress.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('RandomForest_SI_stress_point_' + str(k))
    plt.show()
#%% RANDOM FOREST FI TEMP
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\RandomForest_results\\feature_importances_temp_point'+str(observe_point)+'.csv',
                           index_col = 0)

    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\RF_FI_temp.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('RandomForest_FI_temp_point_' + str(k))
    plt.show()
#%% RANDOM FOREST FI STRESS
points = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for k in range(1,14,1):
    observe_point = k
    points.remove(observe_point) 
    sobol_indices = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\RandomForest_results\\feature_importances_stress_point'+str(observe_point)+'.csv',
                           index_col = 0)

    counter = pd.read_csv('C:\\Users\\defuz\\JupyterLab\\data diplom\\load_time_rounded\\Counters\\RF_FI_stress.csv')
    counter.columns = ['parameters', 'count']
    counter = counter.nlargest(7,'count')['parameters'].to_numpy()
    for parameter in counter:
        Y = sobol_indices.loc[parameter,:].to_numpy()
        X = [5, 10, 20, 30, 40, 50, 150, 1000]
        plt.xlim([11,160])
        plt.plot(X,Y, label = parameter)
        plt.legend(loc = 1, prop={'size': 7})
    plt.title('RandomForest_FI_stress_point_' + str(k))
    plt.show()
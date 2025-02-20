import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from Dataset import STDataset
import os
# from Dataset import Auxility_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml,logging
from models.GWNET import gwnet
from models.MTGNN import MTGNN
# from models import TGCN,ASTGCNCommon,CCRNN,STGCN,AGCRN,STTN,DCRNN
import copy
class CustomStandardScaler:
    def __init__(self, axis=None):
        self.axis = axis
        self.mean = None
        self.std = None

    def fit(self, data):
        if self.axis is None:
            # If axis is not specified, calculate mean and std over the entire data
            self.mean = np.mean(data)
            self.std = np.std(data)
        else:
            # Calculate mean and std along the specified axis
            self.mean = np.mean(data, axis=self.axis)
            self.std = np.std(data, axis=self.axis)

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' method first.")
        
        # Standardize the data using the calculated mean and std
        standardized_data = (data - self.mean) / self.std
        return standardized_data
    
    def inverse_transform(self, standardized_data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' method first.")
        
        # Reverse the standardization process
        original_data = standardized_data * self.std + self.mean
        return original_data
def filter_data(data):
    daily_totals = np.sum(data, axis=(0,2, 3))
    
    # 找到总数不为0的天的索引
    valid_days_idx = np.where(daily_totals > 0)[0]
    
    # 根据索引创建新的data数组
    return data[:,valid_days_idx]

def load_data( config):
    # List all the available files in the data directory


    # Sort the files by name to ensure chronological order



    train_data = np.load(os.path.join('data\\',config['dataset_name'], 'train_data.npy'))
    val_data = np.load(os.path.join('data\\',config['dataset_name'], 'val_data.npy'))
    test_data = np.load(os.path.join('data\\',config['dataset_name'], 'test_data.npy'))
    # print(train_data.shape)
    
    if config['dataset_name'] == 'nycbike':
        
        valid_grid=np.load('data\\nycbike\\valid_grid_bike.npy')
    else:
        
        valid_grid=np.where(train_data.mean(axis=(1,2,3))>2)[0]
    




   
    train_data,val_data,test_data=train_data[valid_grid],val_data[valid_grid],test_data[valid_grid]
   
        
    scaler = CustomStandardScaler()  # Specify the axis over which to calculate mean and std
    scaler.fit(train_data)

    # Standardize the data
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    return train_data, val_data, test_data, scaler,valid_grid

def get_datasets( config):
    # Load and preprocess the data using load_data function
    train_data, val_data, test_data, scaler,valid_gird = load_data( config)

    # Create datasets using the STDataset class
    train_dataset = STDataset(train_data, config)
    val_dataset = STDataset(val_data, config,if_train=False,index=len(train_dataset)//(30*24))
    test_dataset = STDataset(test_data, config,if_train=False,index=len(train_dataset)//(30*24))

    return train_dataset, val_dataset, test_dataset, scaler,valid_gird

def expand_adjacency_matrix(adj_matrix, m):
    n = adj_matrix.shape[0]
    
    if m < n:
        m=n
    
    expanded_adj_matrix = np.zeros((m, m), dtype=int)
    expanded_adj_matrix[:n, :n] = adj_matrix
    
    # Add self-loops
    np.fill_diagonal(expanded_adj_matrix, 1)
    
    return expanded_adj_matrix-np.eye(len(expanded_adj_matrix))

# def align_data(data):
#     # align  data to have the same mean and variance of data[:,:360,:,:]
#     #data (node_number,day_number,24,2)
    
#     target_mean = data[:,-365:].mean()
#     target_std = data[:,:-365:].std()
#     day_number=data.shape[1]
    
#     for i in range(day_number//365):
#         this_mean = data[:,i*365:(i+1)*365,:,:].mean()
#         this_std = data[:,i*365:(i+1)*365,:,:].std()
        
#         data[:,i*365:(i+1)*365,:,:] = (data[:,i*365:(i+1)*365,:,:]-this_mean)/this_std*target_std+target_mean
        
#     return data

# if __name__ == '__main__':
#     # Test the functions
#     with open('models\config2.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     train_data, val_data, test_data, scaler,valid_grid=load_data(config['data_dir'],config)
#     align_data(train_data)

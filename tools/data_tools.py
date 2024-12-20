import numpy as np


from Dataset import STDataset
import os

import numpy as np





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

def load_data(data_dir, config):
    # List all the available files in the data directory
    all_files = os.listdir(data_dir)

    # Sort the files by name to ensure chronological order
    all_files.sort()



    train_data = []
    val_data = []
    test_data = []

    for file_name in all_files:
        # Extract the year and month from the file name (assuming the naming convention is consistent)
        filename_parts = file_name.split('_')
        year = int(filename_parts[1])
        month = int(filename_parts[2][:2])

        # Load the data from the file
        data =filter_data(np.load(os.path.join(data_dir, file_name)))
        if 'taxi' in config['dataset_name']:
            ### beceuse the taxi dataset is of 30min interval, it is needed to reshape it to the 1h interval
            data=data[:-2]
            num_node,day_num,time_in_day,_=data.shape
            data=data.reshape(num_node,day_num,24,2,_).sum(axis=3)
            if year == 2019 :
                # Data for testing (use the entire month)
                test_data.append(data)
            elif year == 2018 and ( month==12):
                # Data for validation (use the last 'val_last_days' days)
                val_data.append(data)
                
            else:
                # Data for training (exclude the testing and validation months)
                train_data.append(data)
        else:
            if year == 2023 :
                # Data for testing (use the entire month)
                test_data.append(data)
            elif year == 2022 and ( month==12):
                # Data for validation (use the last 'val_last_days' days)
                val_data.append(data)
                
            else:
                # Data for training (exclude the testing and validation months)
                train_data.append(data)

    # Concatenate the data along the day_num axis
    train_data =np.concatenate(train_data, axis=1)
    val_data = np.concatenate(val_data, axis=1)
    test_data =np.concatenate(test_data, axis=1)

    valid_grid=np.where(train_data.mean(axis=(1,2,3))>2)[0]

    train_data,val_data,test_data=train_data[valid_grid],val_data[valid_grid],test_data[valid_grid]
    
    # Standardize the data using the provided scaler or StandardScaler
    if 'scaler' in config:
        scaler = config['scaler']
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)
    else:
        # Create and fit a custom scaler
        scaler = CustomStandardScaler()  # Specify the axis over which to calculate mean and std
        scaler.fit(train_data)

        # Standardize the data
        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

    return train_data, val_data, test_data, scaler,valid_grid

def get_datasets(data_dir, config):
    # Load and preprocess the data using load_data function
    train_data, val_data, test_data, scaler,valid_gird = load_data(data_dir, config)

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


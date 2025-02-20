import torch
from torch.utils.data import Dataset
import numpy as np
class STDataset(Dataset):
    def __init__(self, data, config,if_train=True,index=100):
        '''
        data:nparray (num_nodes,day_num,24,2)
        '''

        
        # 根据索引创建新的data数组
        self.data = data
        self.data = self.data.reshape(data.shape[0],-1,2)
        self.input_window = 6
        self.output_window = 1
        self.if_train = if_train
        self.index=index

        self.max_index=(self.data.shape[1] - (self.input_window + self.output_window) + 1)//(30*24)
    def __len__(self):
        return self.data.shape[1] - (self.input_window + self.output_window) + 1

    def __getitem__(self, index):
        
        x = self.data[:, index:index + self.input_window, :].transpose(1,0,2)
        y = self.data[:, index + self.input_window:index + self.input_window + self.output_window, :].reshape(-1,self.output_window,2).transpose(1,0,2)
      
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

    

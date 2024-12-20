import torch 
from torch import nn
from torch.nn import functional as F



class CausalModel_quantile_regress(nn.Module):
    def __init__(self, st_net, config, adj_mx):
        '''
        st_net: (batch_size, input_window, num_nodes, feature_dim_in) -> (batch_size, output_window, num_nodes, feature_dim_out)
        '''
        super(CausalModel_quantile_regress, self).__init__()
        self.config = config.copy()
      
        self.st_net = st_net(self.config, adj_mx)
       
        self.pred_head2 = nn.Linear(16, 2)  # 原点预测值
        self.pred_head_quantiles_up = nn.Linear(16, 2)  # 用于输出分位数
        self.pred_head_quantiles_low = nn.Linear(16, 2)  # 用于输出分位数
        
        self.train_month_num = config['train_months']
        self.device = config['device']
        self.config = config

    def forward(self, x):
        '''
        x: (batch_size, input_window, num_nodes, feature_dim_in)
        '''
        # 通过空间时间网络
        x = self.st_net(x)

        # 点预测
        point_predictions = self.pred_head2(x)  # (batch_size, output_window, num_nodes, feature_dim)

        # 计算分位数预测
        quantile_predictions_up = self.pred_head_quantiles_up(x)  # 计算分位数预测
        quantile_predictions_low = self.pred_head_quantiles_low(x)  # 计算分位数预测
        return point_predictions, quantile_predictions_low,quantile_predictions_up


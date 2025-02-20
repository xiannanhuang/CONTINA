import torch 
from torch import nn
from torch.nn import functional as F
import math

class CausalModel(nn.Module):
    def __init__(self, st_net, config,adj_mx):
        '''
        st-net:(batch_size, input_window, num_nodes, feature_dim_in) ->(batch_size, output_window, num_nodes, feature_dim_out)
        '''
        super(CausalModel, self).__init__()
        self.config=config.copy()
      
        self.st_net = st_net(self.config,adj_mx)
       
        self.pred_head2=nn.Linear(16,2)
     
        self.train_month_num=config['train_months']
        self.device=config['device']
        self.config=config
    
        

    
    def forward(self, x):
        '''
        x:(batch_size, input_window, num_nodes, feature_dim_in)
      
        '''
      
        x=self.st_net(x)
  
        x=self.pred_head2(x) #(batch_size, output_window, num_nodes, feature_dim)
        return x
    
    
    def get_pred_and_feature(self,x):
        feature=self.st_net(x)
        x=self.pred_head2(feature)

        return x,feature
class CausalModel_lp(nn.Module): 
    def __init__(self, st_net, config, adj_mx):
        '''
        st-net: (batch_size, input_window, num_nodes, feature_dim_in) 
                -> (batch_size, output_window, num_nodes, feature_dim_out)
        '''
        super(CausalModel_lp, self).__init__()
        self.config = config.copy()
      
        self.st_net = st_net(self.config, adj_mx)
        
        # 输出头，预测 μ 和 log(b) 两个参数
        self.pred_head_mu = nn.Linear(16, 1)  # 输出位置参数 μ
        self.pred_head_log_b = nn.Linear(16, 1)  # 输出 log(尺度参数 b)
     
        self.train_month_num = config['train_months']
        self.device = config['device']
    
    def forward(self, x):
        '''
        x: (batch_size, input_window, num_nodes, feature_dim_in)
        '''
        x = self.st_net(x)
        
        # 预测 μ 和 log(b)
        mu = self.pred_head_mu(x)  # (batch_size, output_window, num_nodes, 1)
        log_b = self.pred_head_log_b(x)  # (batch_size, output_window, num_nodes, 1)
        b = torch.exp(log_b)  # 将 log(b) 转换为 b，确保 b > 0
        
        return mu, b
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
        
        # self.train_month_num = config['train_months']
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

class CausalModel_var(nn.Module):
    def __init__(self, st_net, config, adj_mx):
        '''
        st-net:(batch_size, input_window, num_nodes, feature_dim_in) -> (batch_size, output_window, num_nodes, feature_dim_out)
        '''
        super(CausalModel_var, self).__init__()
        self.config = config.copy()
      
        self.st_net = st_net(self.config, adj_mx)
        self.dropout= nn.Dropout(p=0.3)
        # Prediction head for mean output
        self.pred_head_mean = nn.Linear(16, 2)  # Adjust the output size as needed
        # Prediction head for variance output
        self.pred_head_variance = nn.Linear(16, 2)  # Same output size for variance
      
        self.train_month_num = config['train_months']
        self.device = config['device']
    
    def forward(self, x):
        '''
        x: (batch_size, input_window, num_nodes, feature_dim_in)
        '''
        # Pass through the spatio-temporal network
        x = self.st_net(x)  # Shape: (batch_size, output_window, num_nodes, feature_dim_out)
        x=self.dropout(x)
        
  
        # Get mean predictions
        mean_pred = self.pred_head_mean(x)  # (batch_size, output_window, num_nodes, feature_dim_out_mean)
        
        # Get variance predictions
        variance_pred = self.pred_head_variance(x)  # (batch_size, output_window, num_nodes, feature_dim_out_variance)
        
        # Optionally apply a softplus activation to ensure variance is positive
        variance_pred = torch.exp(variance_pred)
        
        return mean_pred, variance_pred  # Return both predictions
class CausalModel_MCdropout(nn.Module):
    def __init__(self, st_net, config, adj_mx):
        '''
        st-net:(batch_size, input_window, num_nodes, feature_dim_in) -> (batch_size, output_window, num_nodes, feature_dim_out)
        '''
        super(CausalModel_MCdropout, self).__init__()
        self.config = config.copy()
      
        self.st_net = st_net(self.config, adj_mx)
        self.dropout= nn.Dropout(p=0.3)
        # Prediction head for mean output
        self.pred_head_mean = nn.Linear(16, 2)  # Adjust the output size as needed
        # Prediction head for variance output
        # self.pred_head_variance = nn.Linear(16, 2)  # Same output size for variance
      
        self.train_month_num = config['train_months']
        self.device = config['device']
    
    def forward(self, x):
        '''
        x: (batch_size, input_window, num_nodes, feature_dim_in)
        '''
        # Pass through the spatio-temporal network
        x = self.st_net(x)  # Shape: (batch_size, output_window, num_nodes, feature_dim_out)
        x=self.dropout(x)
  
        # Get mean predictions
        mean_pred = self.pred_head_mean(x)  # (batch_size, output_window, num_nodes, feature_dim_out_mean)
        
        # Get variance predictions
       
 
        return mean_pred
if __name__ == '__main__':
    x=torch.randn(100,1,200,200)
    weight=torch.randn(4,1,3,1)
    x=F.conv2d(x,weight)
    print(x**2)

import torch
import torch.nn as nn

class Causal_CDF_model(nn.Module):
    def __init__(self, st_net, config, adj_mx):
        """
        st-net: (batch_size, input_window, num_nodes, feature_dim_in) -> (batch_size, output_window, num_nodes, feature_dim_out)
        """
        super(Causal_CDF_model, self).__init__()
        self.config = config.copy()
        self.st_net = st_net(self.config, adj_mx)
        
        # 融合分位数信息的预测头
        self.pred_head = nn.Sequential(
            nn.Linear(16 + 1, 16),  # 分位数作为额外输入
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.train_month_num = config['train_months']
        self.device = config['device']

    def forward(self, x, quantiles):
        """
        x: (batch_size, input_window, num_nodes, feature_dim_in)
        quantiles: (batch_size, 1) or scalar, 分位数，表示目标分位点
        返回:
        - preds: (batch_size, output_window, num_nodes, 1), 预测的目标分位点值
        """
        features = self.st_net(x)  # 提取时空特征 (batch_size, output_window, num_nodes, feature_dim)
        
        # 确保 quantiles 是与特征维度匹配的张量
        if isinstance(quantiles, float) or isinstance(quantiles, int):
            quantiles = torch.full_like(features[..., :1], quantiles).to(self.device)  # 广播分位数为张量

        # 拼接分位数信息
        features_with_quantile = torch.cat([features, quantiles], dim=-1)  # (batch_size, ..., feature_dim + 1)

        # 输出目标分位点的预测值
        preds = self.pred_head(features_with_quantile)
        return preds

    def get_pred_and_feature(self, x, quantiles):
        """
        获取预测值及特征。
        返回：
        - preds: 目标分位点值
        - features: 中间特征表示
        """
        features = self.st_net(x)
        
        if isinstance(quantiles, float) or isinstance(quantiles, int):
            quantiles = torch.full_like(features[..., :1], quantiles)

        features_with_quantile = torch.cat([features, quantiles], dim=-1)
        preds = self.pred_head(features_with_quantile)

        return preds, features

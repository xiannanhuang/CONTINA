import torch
from models import causal_model,STGCN
import tools
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch import nn
import yaml
import os
import logging
import random
import tools.data_tools
import time
from itertools import product
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau


def pinball_loss(predictions, targets, quantile):
    """
    计算 Pinball Loss.
    :param predictions: 模型输出的预测值
    :param targets: 实际目标值
    :param quantile: 分位数
    :return: Pinball Loss 值
    """
    errors = targets - predictions
    loss = torch.mean(torch.where(errors >= 0, quantile * errors, (quantile - 1) * errors))
    return loss
def mis_loss(y, u, l, f, rho=0.9):
    """
    MIS loss function for confidence interval estimation.
    
    Parameters:
        y (torch.Tensor): True values.
        u (torch.Tensor): Upper bounds predicted by the model.
        l (torch.Tensor): Lower bounds predicted by the model.
        f (torch.Tensor): Point predictions by the model.
        rho (float): Confidence level (default=0.1).
        
    Returns:
        torch.Tensor: Computed MIS loss.
    """
    # 置信区间长度部分
    interval_length = (u - l)
    
    # 惩罚项 1: 当 y > u 时惩罚
    over_penalty = (y - u).clamp(min=0) * (2 / rho)
    
    # 惩罚项 2: 当 y < l 时惩罚
    under_penalty = (l - y).clamp(min=0) * (2 / rho)
    
    # 预测误差
    prediction_error = torch.abs(y - f)
    
    # 总损失
    loss = interval_length + over_penalty + under_penalty + 0.1*prediction_error
    return loss.mean()
def laplace_nll_loss(y_true, mu, b):
    """
    Computes the negative log-likelihood loss for Laplace distribution.

    Args:
        y_true (Tensor): True target values, shape (batch_size, ...).
        mu (Tensor): Predicted mean (location parameter), shape (batch_size, ...).
        b (Tensor): Predicted scale (scale parameter), shape (batch_size, ...).

    Returns:
        Tensor: Mean negative log-likelihood loss.
    """
    # Ensure b is positive to avoid numerical issues
    b = torch.clamp(b, min=1e-6)  # b > 0

    # Compute absolute error |y_true - mu|
    abs_error = torch.abs(y_true - mu)

    # Compute negative log-likelihood
    nll = abs_error / b + torch.log(2 * b)

    # Return the mean loss
    return torch.mean(nll)
def train(model, train_loader, optimizer, criterion, epoch, device, scale, config):
    start_time = time.time()
    model.train()
    loss_list = []
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        # 进行预测
        point_predictions, lower_bound, upper_bound = model(x)

        # 计算分位数损失
        # 假设你要计算中位数（0.5）和上下 2.5% 的分位数（0.025 和 0.975）
        loss_lower = pinball_loss(lower_bound, y, 0.05)
        loss_upper = pinball_loss(upper_bound, y, 0.95)
        loss_point = pinball_loss(point_predictions, y, 0.5)

        # 总损失可以是三者的平均或其他组合方式
        loss = (loss_lower + loss_upper + loss_point) / 3

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    end_time = time.time()
    print(f'Epoch {epoch + 1} took {end_time - start_time} seconds')
    print(f'Epoch {epoch + 1}, Loss: {np.mean(loss_list) * scale.std}')
    
    return np.mean(loss_list) * scale.std, end_time - start_time
   
    
def evaluate(model, val_loader, criterion, device, scale, quantile_lower=0.05, quantile_upper=0.95):
    loss_list = []
    mae_list = []
    rmse_list = []
    coverage_count = 0
    total_samples = 0
    ci_lengths = []
    
   
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            point_predictions, lower_bound, upper_bound = model(x)
            
            # 计算损失
            loss = criterion(point_predictions, y)
            loss_list.append(loss.item())

            # 计算 MAE 和 RMSE
            mae = torch.mean(torch.abs(point_predictions - y)).item()
            rmse = torch.sqrt(torch.mean((point_predictions - y) ** 2)).item()
            mae_list.append(mae)
            rmse_list.append(rmse)

            # 样本覆盖率
            total_samples += y.size(0)
            coverage_count += ((y >= lower_bound) & (y <= upper_bound)).sum().item()

            # 计算置信区间长度
            ci_length = (upper_bound - lower_bound).mean().item()
            ci_lengths.append(ci_length)
    
                

    # 计算总体指标
    mean_loss = np.mean(loss_list) * scale.std
    mean_mae = np.mean(mae_list)* scale.std
    mean_rmse = np.mean(rmse_list)
    coverage_rate = coverage_count / (total_samples*(x.shape[-1]*x.shape[-2]))
    mean_ci_length = np.mean(ci_lengths)

    print(f'Validation Loss: {mean_loss}')
    print(f'MAE: {mean_mae}')
    print(f'RMSE: {mean_rmse}')
    print(f'Coverage Rate: {coverage_rate:.4f}')
    print(f'Average Confidence Interval Length: {mean_ci_length}')

    return mean_loss, mean_mae, mean_rmse, coverage_rate, mean_ci_length


def predict(model, test_loader, device):
    model.eval()
    true_values = []
    predicted_means = []
    predicted_lower_bounds = []
    predicted_upper_bounds = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            point_predictions, lower_bound, upper_bound = model(x)

            # 记录每个样本的真实值和预测值
            true_values.extend(y.cpu().numpy())
            predicted_means.extend(point_predictions.cpu().numpy())
            predicted_lower_bounds.extend(lower_bound.cpu().numpy())
            predicted_upper_bounds.extend(upper_bound.cpu().numpy())

    return true_values, predicted_means, predicted_lower_bounds, predicted_upper_bounds
def dict2srt(dictory):
    return ' '.join([f'{key}_{value}' for key,value in dictory.items() if '/' not in str(value)])





for _ in range(1):
    train_loss_list=[]
    val_loss_list=[]
    test_loss_list=[]
    log_file=os.path.join('logs','train.log')
    
    
    with open('models\config.yaml', 'r') as f:
        config = yaml.safe_load(f)



    train_data, val_data, test_data, scaler,valid_grid=tools.data_tools.get_datasets(config['data_dir'],config)
    adj_mx =  tools.data_tools.expand_adjacency_matrix(np.load(config['adj_mx_file']), config['num_nodes'])[valid_grid][:,valid_grid]
    config['num_nodes']=len(valid_grid)
    config['train_months']=train_data.data.shape[1]//(30*24)+1
    device=config.get('device')
    print(config)
    
        
    
    model = causal_model.CausalModel_quantile_regress(STGCN.STGCN,config,adj_mx)
        


    model.to(device)

    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.L1Loss()

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    for epoch in range(100):
        train_loss, train_time = train(model, train_loader, optimizer, criterion, epoch, device, scaler, config)
        
        # 使用新的 evaluate 函数
        val_loss, val_mae, val_rmse, coverage_rate, mean_ci_length = evaluate(model, val_loader, criterion, device, scaler)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}, '
                    f'MAE: {val_mae:.3f}, RMSE: {val_rmse:.3f}, Coverage Rate: {coverage_rate:.3f}, '
                    f'Mean CI Length: {mean_ci_length:.3f}, train_time: {train_time:.3f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_save_path = f'{config["model_dir"]}/epoch{epoch+1}_val_loss{val_loss:.4f}'+'.pth'
            torch.save(model.state_dict(), best_model_save_path)
            print('Model saved!')
        
        # Adjust learning rate if validation loss does not improve
        scheduler.step(val_loss)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


    # Load the best model
    model.load_state_dict(torch.load(best_model_save_path))
    save_dir = f'final_models/{config["dataset_name"]}'
    os.makedirs(save_dir, exist_ok=True)

    # Define paths for saving the model and config
    best_model_save_path = os.path.join(save_dir, 'model.pth')
    model_config_save_path = os.path.join(save_dir, 'config.yaml')

    # Save the configuration to a YAML file
    with open(model_config_save_path, 'w') as f:
        yaml.dump(config, f)

    # Save the model state dictionary
    torch.save(model.state_dict(), best_model_save_path)

    # 再次评估测试数据（如果需要）
    test_loss, test_mae, test_rmse, coverage_rate, mean_ci_length = evaluate(model, test_loader, criterion, device, scaler) 
    
    print(f' Test Loss: {test_loss}, '
                    f'MAE: {test_mae}, RMSE: {test_rmse}, Coverage Rate: {coverage_rate:.4f}, '
                    f'Mean CI Length: {mean_ci_length}, train_time: {train_time}')
 

import torch
from models import DCRNN,causal_model,AGCRN,GWNET,STGCN,MTGNN
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
import tools.train_tools
from tqdm import tqdm
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
def train(model, train_loader, optimizer, criterion, epoch, device, scale, config,mode='quantile'):
    start_time = time.time()
    model.train()
    loss_list = []
    if mode == 'laplace':
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # Model outputs μ and b for Laplace distribution
            mu, b = model(x)

            # Compute loss using Laplace NLL
            loss = laplace_nll_loss(y, mu, b)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            end_time = time.time()
        print(f'Epoch {epoch + 1} took {end_time - start_time} seconds')
        print(f'Epoch {epoch + 1}, Loss: {np.mean(loss_list) * scale.std}')
        
        return np.mean(loss_list) * scale.std, end_time - start_time
    if mode=='quantile':
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
            for x, y in tepoch:
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
    if mode=='any_quantile':
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            for i in range(3):
                optimizer.zero_grad()
                quantile=random.random()
                # 进行预测
                point_predictions= model(x,quantile)

                # 计算分位数损失
                
                loss = pinball_loss(point_predictions, y, quantile)
            

                loss.backward()
                optimizer.step()
            loss_list.append(loss.item())

        end_time = time.time()
        print(f'Epoch {epoch + 1} took {end_time - start_time} seconds')
        print(f'Epoch {epoch + 1}, Loss: {np.mean(loss_list) * scale.std}')
        
        return np.mean(loss_list) * scale.std, end_time - start_time
    
    if mode=='MC_dropout':
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            # 进行预测
            point_predictions=model(x)
            # 计算损失
            loss = criterion(point_predictions, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        end_time = time.time()
        print(f'Epoch {epoch + 1} took {end_time - start_time} seconds')
        print(f'Epoch {epoch + 1}, Loss: {np.mean(loss_list) * scale.std}')
    elif mode == 'MIS':
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
            for x, y in tepoch:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                # 进行预测：输出 point_predictions, lower_bound, upper_bound
                point_predictions, lower_bound, upper_bound = model(x)
                
                # 使用 MIS 损失
                loss = mis_loss(y, upper_bound, lower_bound, point_predictions, rho=config.get("rho", 0.1))
                
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            
        end_time = time.time()
        print(f'Epoch {epoch + 1} took {end_time - start_time} seconds')
        print(f'Epoch {epoch + 1}, MIS Loss: {np.mean(loss_list)}')


        return np.mean(loss_list) * scale.std, end_time - start_time
    
def evaluate(model, val_loader, criterion, device, scale, quantile_lower=0.05, quantile_upper=0.95,mode='quantile'):
    loss_list = []
    mae_list = []
    rmse_list = []
    coverage_count = 0
    total_samples = 0
    ci_lengths = []
    if mode == 'laplace':
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                # Model outputs μ and b
                mu, b = model(x)

                # Compute Laplace NLL loss
                loss = laplace_nll_loss(y, mu, b)
                loss_list.append(loss.item())

                # Compute MAE and RMSE
                mae = torch.mean(torch.abs(mu - y)).item()
                rmse = torch.sqrt(torch.mean((mu - y) ** 2)).item()
                mae_list.append(mae)
                rmse_list.append(rmse)

                # Confidence interval [μ - z*b, μ + z*b] based on Laplace
            
                lower_bound = mu - 2.3026 * b
                upper_bound = mu + 2.3026 * b

                # Compute coverage
                total_samples += y.size(0)
                coverage_count += ((y >= lower_bound) & (y <= upper_bound)).sum().item()

                # Compute CI length
                ci_length = (upper_bound - lower_bound).mean().item()
                ci_lengths.append(ci_length)
    if mode=='quantile' or mode=='MIS':
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
    if mode=='MC_dropout':
        with torch.no_grad():
           for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            # 进行多次前向传播（MC Dropout）
            point_predictions_list = []
            for _ in range(20):  # num_mc_samples 是 MC Dropout 的样本数
                point_predictions = model(x)
                point_predictions_list.append(point_predictions)

            # 计算预测均值和方差
            point_predictions_mean = torch.mean(torch.stack(point_predictions_list), dim=0)
            point_predictions_var = torch.var(torch.stack(point_predictions_list), dim=0)

            # 损失计算
            loss = criterion(point_predictions_mean, y)
            loss_list.append(loss.item())

            # MAE 和 RMSE 计算
            mae = torch.mean(torch.abs(point_predictions_mean - y)).item()
            rmse = torch.sqrt(torch.mean((point_predictions_mean - y) ** 2)).item()
            mae_list.append(mae)
            rmse_list.append(rmse)

            # 样本覆盖率
            total_samples += y.size(0)
            coverage_count += ((y >= point_predictions_mean - 1.645 * torch.sqrt(point_predictions_var)) & 
                               (y <= point_predictions_mean + 1.645 * torch.sqrt(point_predictions_var))).sum().item()

            # 置信区间长度
            ci_length = (2 * 1.645 * torch.sqrt(point_predictions_var)).mean().item()  # 95% 置信区间
            ci_lengths.append(ci_length)
    if mode=='any_quantile':
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                low=quantile_lower
                up=quantile_upper
                mean=0.5
                point_predictions= model(x,mean)
                lower_bound, upper_bound = model(x,low), model(x,up)
                
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

def test(model, test_loader, criterion, device,scale):
    model.eval()
    loss_list = []
    mae_list = []
    rmse_list = []
    coverage_count = 0
    total_samples = 0
    ci_lengths = []

    with torch.no_grad():
        for x, y in test_loader:
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
mode='MIS'
import itertools
learning_rates = [0.005]
batch_sizes = [128]
epoch_nums = [60]
hyperparameter_combinations = list(itertools.product(learning_rates, batch_sizes, epoch_nums))

# 确保日志和模型目录存在
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

for lr, batch_size, num_epochs in hyperparameter_combinations:
    for model_name in ['DCRNN','STGCN','MTGNN','gwnet']:  # 可以扩展为多个模型
        train_loss_list = []
        val_loss_list = []
        test_loss_list = []

        with open('models/nyctaxi_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新超参数
        config['model_name'] = model_name
        config['learning_rate'] = lr
        config['batch_size'] = batch_size
        config['num_epochs'] = num_epochs
        dataset=config['dataset_name']

        # 数据加载与预处理
        train_data, val_data, test_data, scaler, valid_grid = tools.data_tools.get_datasets(config)
        adj_mx = np.load(config['adj_mx_file']), config['num_nodes'][valid_grid][:, valid_grid]
        config['num_nodes'] = len(valid_grid)
        config['train_months'] = train_data.data.shape[1] // (30 * 24) + 1

        config['device'] = 'cuda:1' 
        device = torch.device(config['device'])
        print(f"Training with config: {config}")

        # 初始化模型
        if config['model_name'] == 'STGCN':
            model = causal_model.CausalModel_quantile_regress(STGCN.STGCN, config, adj_mx)
        if config['model_name'] == 'DCRNN':
            model = causal_model.CausalModel_quantile_regress(DCRNN.DCRNN, config, adj_mx)
        if config['model_name'] == 'gwnet':
            model = causal_model.CausalModel_quantile_regress(GWNET.gwnet, config, adj_mx)
        if config['model_name'] == 'MTGNN':
            model = causal_model.CausalModel_quantile_regress(MTGNN.MTGNN, config, adj_mx)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.L1Loss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # 数据加载器
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

        best_val_loss = float('inf')
        for epoch in range(config['num_epochs']):
            train_loss, train_time = train(model, train_loader, optimizer, criterion, epoch, device, scaler, config,mode=mode)

            val_loss, val_mae, val_rmse, coverage_rate, mean_ci_length = evaluate(model, val_loader, criterion, device, scaler,mode=mode)

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}, "
                  f"MAE: {val_mae:.3f}, RMSE: {val_rmse:.3f}, Coverage Rate: {coverage_rate:.3f}, "
                  f"Mean CI Length: {mean_ci_length:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = f'checkpoints/{model_name}_lr{lr}_bs{batch_size}_epochs{num_epochs}_epoch{epoch+1}_val_loss{val_loss:.4f}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

            scheduler.step(val_loss)

            test_loss, test_mae, test_rmse, coverage_rate, mean_ci_length = evaluate(model, test_loader, criterion, device, scaler)

            print(f"Test Loss: {test_loss:.3f}, "
                  f"MAE: {test_mae:.3f}, RMSE: {test_rmse:.3f}, Coverage Rate: {coverage_rate:.3f}, "
                  f"Mean CI Length: {mean_ci_length:.3f}")

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_mae)

        # 保存最终模型和配置
        save_dir = f"final_models/{config['dataset_name']}/{model_name}/{mode}/lr{lr}_bs{batch_size}_epochs{num_epochs}_"
        # save_dir = f"final_models/nycbike/{model_name}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

        # 保存损失记录
        np.save(os.path.join(save_dir, 'loss.npy'), np.array([train_loss_list, val_loss_list, test_loss_list]))
        print(f"Training completed for model {model_name} with lr={lr}, batch_size={batch_size}, epochs={num_epochs}")

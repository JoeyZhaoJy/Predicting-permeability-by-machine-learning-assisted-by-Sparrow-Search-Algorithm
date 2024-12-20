# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:57:35 2024

@author: PC
"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
import time

import pandas as pd
# import sparrow as spa
import matplotlib.pyplot as plt



path = r'G:\文章成果\期刊论文\水合物储层渗透率\数据\绝对渗透率.xlsx'
sheetname = 'Sheet1 (2)' 
num =  800
data_frame = pd.read_excel(path, str(sheetname))
str1 = data_frame.columns
data_frame = np.array(data_frame)

# 标准化数据
scaler = StandardScaler().fit(data_frame)
data_frame = scaler.transform(data_frame)

data = data_frame[:num]
x_train = data[:, : -3]
y_train = data[:, -3:]

data = data_frame[num:1000]
x_test = data[:, : -3]
y_test = data[:, -3:]

model = RandomForestRegressor(n_estimators=86,max_features=8,
                                   bootstrap=True,
                                   )
forest_reg = MultiOutputRegressor(model)

n = 15
from sklearn.model_selection import cross_validate, KFold
# 实例化交叉验证方式 shuffle是否打乱数据
cv = KFold(n_splits=n, shuffle=True,  random_state=18) #
result = cross_validate(forest_reg, # 评估器
                        x_train, x_train, # 数据
                        cv=n # 交叉验证模式
                        ,scoring='neg_mean_squared_error' # r2 'neg_mean_squared_error' # 评估指标mse(负值)
                        # 以上参数同cross_val_score()
                        # 以下cross_validate()特有参数
                        ,return_train_score=True # 返回训练集交叉验证分数
                        ,verbose=True # 打印进程
                        ,n_jobs=8 # 线程数 -1表示调用全部线程
                       )

'''
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.6s finished
'''
# 画图观察
plt.rcParams['font.family']=' Times New Roman'
plt.figure(figsize=(8,6), dpi=80)
plt.plot(range(1, n+1), abs(result['test_score']), color='green',label='Test')
plt.plot(range(1, n+1), abs(result['train_score']), color='red',label='Train')
plt.xticks(range(1, n+1))
plt.xlabel('CVcounts', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.legend()
plt.show()





#%%
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
# from sklearn.multioutput import MultiOutputRegressor
import time
from sklearn.model_selection import cross_validate, KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

import pandas as pd
# import sparrow as spa
import matplotlib.pyplot as plt

path = r'G:\文章成果\期刊论文\水合物储层渗透率\数据\*.xlsx'
sheetname = '*' 
num =  int(2420*0.8)
data_frame = pd.read_excel(path, str(sheetname))
str1 = data_frame.columns
data_frame = np.array(data_frame)

# 标准化数据
scaler = StandardScaler().fit(data_frame)
data_frame = scaler.transform(data_frame)

data = data_frame[:num]
x_train = data[:, : -1]
y_train = data[:, -1]

data = data_frame[num:2420]
x_test = data[:, : -1]
y_test = data[:, -1]
#%%
class DNNNet(nn.Module):
    def __init__(self, input_features, hidden_nodes0, hidden_nodes1, hidden_nodes2, output_class):
        super(DNNNet,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_features, hidden_nodes0),
            nn.LeakyReLU(),
            nn.Linear(hidden_nodes0, hidden_nodes1),
            nn.LeakyReLU(),
            nn.Linear(hidden_nodes1, hidden_nodes2),
            nn.LeakyReLU(),
            nn.Linear(hidden_nodes2, output_class),
            # nn.LeakyReLU(),
            )

    def forward(self, x):
        x = self.main(x)

        return x


alpha = 7.5159e-3 # 学习率
num_epochs = 86 #迭代次数
batch_size = 23 # batchsize
hidden_nodes0 = 70 #第一隐含层神经元
hidden_nodes1 = 68 #第二隐含层神经元
hidden_nodes2 = 53 #第二隐含层神经元

input_features = x_train.shape[1]
output_class =  1

k = 15 # k-fold

model_DNN = DNNNet(input_features, hidden_nodes0, hidden_nodes1, hidden_nodes2,output_class).to('cuda')
# print(model_DNN)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model_DNN.parameters(),lr= alpha)

splits=KFold(n_splits=k,shuffle=True,random_state=1)
foldperf={}
history = {'train_loss': [], 'test_loss': []}

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_frame)))):
    print('Fold {}'.format(fold + 1))
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(data_frame, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(data_frame, batch_size=batch_size, sampler=test_sampler)
    for epoch in range(num_epochs):
        model_DNN.train()
        train_loss = 0
        for data in train_loader:
            data = torch.as_tensor(data, dtype = torch.float).to('cuda')
            x = data[:,:-1]
            y = data[:,-1]
            optimizer.zero_grad()
            output = model_DNN(x)
            loss = criterion(y,output)
            train_loss += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
        
        model_DNN.eval()
        test_loss = 0
        for data in test_loader:
            data = torch.as_tensor(data, dtype = torch.float).to('cuda')
            x = data[:,:-1]
            y = data[:,-1]
            with torch.no_grad():                   # 取消梯度计算(加快运行速度)
                pred = model_DNN(x)                     # 前向计算
                mse_loss = criterion(y, pred)  # 计算损失
                test_loss += mse_loss.detach().cpu().numpy()

    train_loss = train_loss / len(train_loader.sampler)
    test_loss = test_loss / len(test_loader.sampler)
    print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f}".format(epoch + 1, num_epochs, 
                                                                             train_loss, test_loss,))
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    # foldperf['fold{}'.format(fold+1)] = history 
    # torch.save(model,'k_cross_CNN.pt')


# 画图观察
plt.rcParams['font.family']=' Times New Roman'
plt.figure(figsize=(8,6), dpi=80)
plt.plot(range(1, k+1), history['test_loss'], color='green',label='Test')
plt.plot(range(1, k+1), history['train_loss'], color='red',label='Train')
plt.xticks(range(1, k+1))
plt.xlabel('CVcounts', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.legend()
plt.show()










# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:26:16 2023

@author: Joey
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:46:02 2023
ssa优化lstm超参数
https://github.com/Knight0111/SSA-LSTM/blob/master/3.ssa%E4%BC%98%E5%8C%96lstm%E8%B6%85%E5%8F%82%E6%95%B0.py
https://github.com/changliang5811/SSA_python/blob/master/function.py
https://github.com/LucXiong/Swarm-intelligence-optimization-algorithm/blob/main/SSA2020.py
https://blog.csdn.net/weixin_44252015/article/details/126206762
@author: HP
"""
import warnings
warnings.filterwarnings("ignore")
import os

import time
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random
#import tensorflow as tf#tensorflow1.x环境就用这个

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from scipy.io import savemat,loadmat
from matplotlib.pyplot import MultipleLocator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
start = time.time()

#%%
'''
进行适应度计算,以验证集均方差为适应度函数，目的是找到一组超参数 使得网络的误差最小
# '''

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
        # self.layer1 = nn.Linear(input_features, hidden_nodes0)
        # # self.act1 = nn.LeakyReLU()
        # self.layer2 = nn.Linear(hidden_nodes0, hidden_nodes)
        # # self.act1 = nn.LeakyReLU()
        # self.layer3 = nn.Linear(hidden_nodes, output_class)
        # self.layer4 = nn.Linear(input_features, hidden_nodes0),
        


    def forward(self, x):
        # x = x #.view(-1,2) # Flattern the (n,1,28,28) to (n,784)
        # x = F.leaky_relu(self.layer1(x))
        # x = F.leaky_relu(self.layer2(x))
        # x = self.layer3(x)
        x = self.main(x)

        return x

class loaddataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self,):
        return len(self.x)
    
    def __getitem__(self, i):
        X = self.x[i]
        Y = self.y[i]
        return X, Y
    
#%%
def fun(pop, P,T,Pt,Tt):
    alpha = pop[0]# 学习率
    num_epochs = int(pop[1])#迭代次数
    batch_size = int(pop[2]) # batchsize
    hidden_nodes0 = int(pop[3])#第一隐含层神经元
    hidden_nodes1 = int(pop[4])#第二隐含层神经元
    hidden_nodes2 = int(pop[5])#第二隐含层神经元

    input_features = P.shape[1]
    output_class =  T.shape[1] # 34输入 24输出
     
    
    # print(input_features,hidden_nodes0,hidden_nodes,output_class)
    # print(P.shape,T.shape)
    
    model_DNN = DNNNet(input_features, hidden_nodes0, hidden_nodes1, hidden_nodes2,output_class).to('cuda')
    # print(model_DNN)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model_DNN.parameters(),lr= alpha)
    train_dataset = loaddataset(P,T)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

    model_DNN.train()
    for i in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = torch.as_tensor(data, dtype = torch.float).to('cuda')
            target = torch.as_tensor(target, dtype =torch.float).to('cuda')
            optimizer.zero_grad()
            output = model_DNN(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
        
    test_dataset = loaddataset(Pt,Tt)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    model_DNN.eval()
    test_loss = 0
    for data, target in test_loader:
        x = torch.as_tensor(data, dtype = torch.float).to('cuda')
        y = torch.as_tensor(target, dtype = torch.float).to('cuda')     
        with torch.no_grad():                   # 取消梯度计算(加快运行速度)
            pred = model_DNN(x)                     # 前向计算
            mse_loss = criterion(y, pred)  # 计算损失
            test_loss += mse_loss


    return test_loss.cpu().numpy()

  
def boundary(pop,lb,ub):
    # 防止粒子跳出范围,除学习率之外 其他的都是整数
    pop=[int(pop[i]) if i>0 else pop[i] for i in range(len(lb))]
    for i in range(len(lb)):
        if pop[i]>ub[i] or pop[i]<lb[i]:
            if i==0:
                pop[i] = (ub[i]-lb[i])*np.random.rand()+lb[i]
            else:
                pop[i] = np.random.randint(lb[i],ub[i])
    return pop
def SSA(maxiter, pop, dim, lb, ub, P,T,Pt,Tt):
    M=maxiter
    pop=pop
    P_percent=0.2
    dim = dim #搜索维度,第一个是学习率[0.001 0.01]
    #第二个是迭代次数[10-100] 
    #第三和第四个是隐含层节点数[1-100]
    Lb= lb
    Ub= ub
    #M 迭代次数
    #pop 麻雀种群数量
    #dim 寻优维度
    #P_percent 麻雀在生产者的比例
    pNum = round( pop *  P_percent )#pNum是生产者
    x=np.zeros((pop,dim))
    # print("len",len(x[0]))
    fit=np.zeros((pop,1))
    # 种群初始化
    for i in range(pop):    
        for j in range(dim):    
            if j==0:#学习率是小数 其他的是整数
                x[i][j] = (Ub[j]-Lb[j])*np.random.rand()+Lb[j]
            else:
                x[i][j] = np.random.randint(Lb[j],Ub[j])
        
        fit[ i ]  = fun( x[ i, : ],P,T,Pt,Tt )
    pFit = fit.copy()
    pX = x.copy()
    fMin=np.min( fit )
    bestI=np.argmin( fit )
    bestX = x[bestI, : ].copy()
    Convergence_curve=np.zeros((M,))
    result=np.zeros((M,dim))

    for t in range(M):
        sortIndex = np.argsort( pFit.reshape(-1,) ).reshape(-1,)
        fmax=np.max( pFit )
        B=np.argmax( pFit )
        worse= x[B,:].copy()
        r2=np.random.rand()
        ## 这一部分为发现者（探索者）的位置更新
        if r2<0.8:#%预警值较小，说明没有捕食者出现
            for i in range(pNum):#r2小于0.8时发现者改变位置
                r1=np.random.rand()
                x[sortIndex[i],:]=pX[sortIndex[i],:]*np.exp(-i/(r1*M))
                x[sortIndex[i],:]=boundary(x[sortIndex[i],:],Lb,Ub)
                temp=fun( x[ sortIndex[ i ], : ],P,T,Pt,Tt )
                fit[ sortIndex[ i ] ] = temp# 计算新的适应度值
        else:#预警值较大，说明有捕食者出现威胁到了种群的安全，需要去其它地方觅食
            for i in range(pNum):#r2大于0.8时发现者改变位置
                r1=np.random.rand()
                x[sortIndex[i],:]=pX[sortIndex[i],:] + np.random.normal()*np.ones((1,dim))
                x[sortIndex[i],:]=boundary(x[sortIndex[i],:],Lb,Ub)
                fit[ sortIndex[ i ] ] = fun( x[ sortIndex[ i ], : ],P,T,Pt,Tt )# 计算新的适应度值
        bestII=np.argmin( fit )
        bestXX = x[ bestII, : ].copy()
        
        ##这一部分为加入者（追随者）的位置更新
        for i in range(pNum + 1,pop):#剩下的个体变化
            A=np.floor(np.random.rand(1,dim)*2)*2-1
            if i > pop/2: #这个代表这部分麻雀处于十分饥饿的状态（因为它们的能量很低，也是是适应度值很差），需要到其它地方觅食
                x[ sortIndex[i ], : ]=np.random.normal()*np.exp((worse-pX[sortIndex[ i ], : ])/(i**2))
            else:#这一部分追随者是围绕最好的发现者周围进行觅食，其间也有可能发生食物的争夺，使其自己变成生产者
                
                x[ sortIndex[ i ], : ]=bestXX+np.abs( pX[ sortIndex[ i ], : ]-bestXX).dot(A.T*(A*A.T)**(-1))*np.ones((1,dim))
            x[sortIndex[ i ], : ] = boundary( x[ sortIndex[ i ], : ],Lb,Ub)#判断边界是否超出
            fit[ sortIndex[ i ] ] = fun( x[ sortIndex[ i ], : ],P,T,Pt,Tt )#计算适应度值        
        
        #这一部分为意识到危险（注意这里只是意识到了危险，不代表出现了真正的捕食者）的麻雀的位置更新
        c=random.sample(range(sortIndex.shape[0]),sortIndex.shape[0])#这个的作用是在种群中随机产生其位置（也就是这部分的麻雀位置一开始是随机的，意识到危险了要进行位置移动，
        b=sortIndex[np.array(c)[0:round(pop*0.2)]].reshape(-1,)
        for j in range(b.shape[0]):
            if pFit[sortIndex[b[j]]]> fMin:#处于种群外围的麻雀的位置改变
                x[ sortIndex[ b[j] ], : ]=bestX+np.random.normal(1,dim)*(np.abs( pX[ sortIndex[ b[j] ], : ]  -bestX))

            else: #处于种群中心的麻雀的位置改变
                x[ sortIndex[ b[j] ], : ] =pX[ sortIndex[ b[j] ], : ] + (2*np.random.rand()-1)*(np.abs(pX[ sortIndex[ b[j] ], : ]-worse)) / (pFit[sortIndex[b[j]]]-fmax+1e-50)
            x[ sortIndex[b[j] ], : ] = boundary( x[ sortIndex[b[j]], : ],Lb,Ub)
            fit[ sortIndex[b[j] ] ] = fun( x[ sortIndex[b[j] ], : ], P,T,Pt,Tt )#计算适应度值 
        
        # 这部分是最终的最优解更新
        for i in range(pop):
            if  fit[ i ] < pFit[ i ] :
                pFit[ i ] = fit[ i ].copy()
                pX[ i, : ] = x[ i, : ].copy()
        
            if  pFit[i ] < fMin:
                fMin= pFit[ i ,0].copy()
                bestX = pX[ i, :].copy()
        result[t,:]=bestX
        print(t+1,fMin,[int(bestX[i]) if i>0 else bestX[i] for i in range(len(Lb))])    

        Convergence_curve[t]=fMin
    return bestX,Convergence_curve,result

# In[] 加载数据
print("SSA-DNN参数优化开始：")
time_pre = time.time()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time_pre))))

maxiter = 50 # 迭代次数
pop = 30 # pop 麻雀种群数量
dim = 6 #dim 寻优维度
#第一个是学习率[0.001 0.01]
#第二个是迭代次数[10-100] 
#第三个是批大小[10-100] 
#第四和第五个是隐含层节点数[1-100]
lb=[0.000001, 10,  10,  1,   1,   1]
ub=[0.01 , 200, 100, 100, 100, 100]

path = r'G:\文章成果\期刊论文\水合物储层渗透率\数据\1 第4组.xlsx'
sheetname = 'Sheet2' 
num =  2419
data_frame = np.array(pd.read_excel(path, str(sheetname)))
# num = 240 # int(len(data_frame) * 0.8)
data = data_frame[:num]
# 标准化数据
scaler = StandardScaler().fit(data)
data = scaler.transform(data)
feature = data[:, : 3]
target = data[:, -1]
    
train_data, test_data, train_label, test_label = train_test_split(feature, target, test_size=0.2, random_state = 1)


best,trace,result=SSA(maxiter, pop, dim, lb, ub,train_data,train_label.reshape(-1,1),test_data,test_label.reshape(-1,1))

time_now = time.time()
time_gap = (time_now - time_pre) / 60
print('运行时间：%d min' % time_gap)
# savemat('结果/ssa_para.mat',{'trace':trace,'best':best,'result':result})
# In[]
# trace=loadmat('结果/ssa_para.mat')['trace'].reshape(-1,)
# result=loadmat('结果/ssa_para.mat')['result']
#%%
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize=(5, 4), dpi=800)
plt.plot(trace)
plt.title('fitness curve',fontproperties = 'Times New Roman', fontsize=12)
plt.xlabel('iteration',fontproperties = 'Times New Roman', fontsize=12)
plt.ylabel('fitness value',fontproperties = 'Times New Roman', fontsize=12)
my_x_ticks = np.arange(0, 21, 5)
plt.xticks(my_x_ticks)
plt.xticks(fontproperties = 'Times New Roman', fontsize=12)
plt.yticks(fontproperties = 'Times New Roman', fontsize=12)
# print("fitness curve",trace)
# plt.savefig("ssa_lstm图片保存/fitness curve.png")

plt.figure(figsize=(5, 4), dpi=800)
plt.plot(result[:,0])
plt.title('learning rate optim',fontproperties = 'Times New Roman', fontsize=12)
plt.xlabel('iteration',fontproperties = 'Times New Roman', fontsize=12)
plt.ylabel('learning rate value',fontproperties = 'Times New Roman', fontsize=12)
my_x_ticks = np.arange(0, 21, 5)
plt.xticks(my_x_ticks)
plt.xticks(fontproperties = 'Times New Roman', fontsize=12)
plt.yticks(fontproperties = 'Times New Roman', fontsize=12)
# print("learning rate optim",result[:,0])
# plt.savefig("ssa_lstm图片保存/learning rate optim.png")


plt.figure(figsize=(5, 4), dpi=800)
plt.plot(result[:,1])
plt.title('itration optim',fontproperties = 'Times New Roman', fontsize=12)
plt.xlabel('iteration',fontproperties = 'Times New Roman', fontsize=12)
plt.ylabel('itration value',fontproperties = 'Times New Roman', fontsize=12)
my_x_ticks = np.arange(0, 21, 5)
plt.xticks(my_x_ticks)
plt.xticks(fontproperties = 'Times New Roman', fontsize=12)
plt.yticks(fontproperties = 'Times New Roman', fontsize=12)
# print("itration optim",result[:,1])
# plt.savefig("ssa_lstm图片保存/itration optim.png")


plt.figure(figsize=(5, 4), dpi=800)
plt.plot(result[:,2])
plt.title('first hidden nodes optim',fontproperties = 'Times New Roman', fontsize=12)
plt.xlabel('iteration',fontproperties = 'Times New Roman', fontsize=12)
plt.ylabel('first hidden nodes value',fontproperties = 'Times New Roman', fontsize=12)
my_x_ticks = np.arange(0, 21, 5)
plt.xticks(my_x_ticks)
plt.xticks(fontproperties = 'Times New Roman', fontsize=12)
plt.yticks(fontproperties = 'Times New Roman', fontsize=12)
# print("first hidden nodes optim",result[:,2])
# plt.savefig("ssa_lstm图片保存/first hidden nodes optim.png")


plt.figure(figsize=(5, 4), dpi=800)
plt.plot(result[:,3])
plt.title('second hidden nodes optim',fontproperties = 'Times New Roman', fontsize=12)
plt.xlabel('iteration',fontproperties = 'Times New Roman', fontsize=12)
plt.ylabel('second hidden nodes value',fontproperties = 'Times New Roman', fontsize=12)
my_x_ticks = np.arange(0, 21, 5)
plt.xticks(my_x_ticks)
plt.xticks(fontproperties = 'Times New Roman', fontsize=12)
plt.yticks(fontproperties = 'Times New Roman', fontsize=12)
# print("second hidden nodes optim",result[:,3])
# plt.savefig("ssa_lstm图片保存/second hidden nodes optim.png")
plt.show()


print(best)


#%%
path = r'G:\文章成果\期刊论文\水合物储层渗透率\数据\静态孔隙结构参数.xlsx'
sheetname = 'Sheet6' 
num =  1000
data_frame = np.array(pd.read_excel(path, str(sheetname)))
# data_frame = data_frame[:,1:4]
# 标准化数据
scaler = StandardScaler().fit(data_frame)
data_frame = scaler.transform(data_frame)

data = data_frame[:num]
x_train = data[:, : 3]
y_train = data[:, -1]#.reshape(-1,1)

data = data_frame[num:]
x_test = data[:, : 3]
y_test = data[:, -1]#.reshape(-1,1)

input_features = x_train.shape[1]
output_class = y_train.shape[1]


#%%
alpha = best[0]# 学习率
num_epochs = int(best[1])#迭代次数
batch_size = int(best[2])# batchsize
hidden_nodes0 = int(best[3])#第一隐含层神经元
hidden_nodes1 = int(best[4])#第二隐含层神经元
hidden_nodes2 = int(best[5])#第二隐含层神经元

# input_features = x_train.shape[1]
# output_class =  y_train.shape[1] # 34输入 24输出


model_DNN = DNNNet(input_features, hidden_nodes0, hidden_nodes1, hidden_nodes2, output_class).to('cuda')
# print(model_DNN)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model_DNN.parameters(),lr= alpha)
train_dataset = loaddataset(x_train,y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)

model_DNN.train()
for i in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.as_tensor(data, dtype = torch.float).to('cuda')
        target = torch.as_tensor(target, dtype =torch.float).to('cuda')
        optimizer.zero_grad()
        output = model_DNN(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
    

model_DNN.eval()
x = torch.as_tensor(x_test, dtype = torch.float).to('cuda')
y = torch.as_tensor(y_test, dtype = torch.float).to('cuda')     
with torch.no_grad():                   # 取消梯度计算(加快运行速度)
    testPredict = model_DNN(x)                     # 前向计算

realdata = np.concatenate((x_test, y_test),axis = 1) # data_frame[num:, 3:-1]
realdata = scaler.inverse_transform(realdata)
testPredict = testPredict.detach().cpu().numpy()
predata = np.concatenate((x_test,  testPredict),axis = 1) # data_frame[num:, 3:-1],
predata = scaler.inverse_transform(predata)

#%
fig = plt.figure(dpi = 300)
# plt.scatter(realdata[:,0],realdata[:,-1], label ='Origin')
plt.scatter(predata[:,9],realdata[:,9], label ='Kx')
plt.scatter(predata[:,10],realdata[:,10], label ='Ky')
plt.scatter(predata[:,11],realdata[:,11], label ='Kz')
plt.xlabel('Prediction')
plt.ylabel('LBM')
plt.legend(loc='upper right')

fig = plt.figure(dpi = 300)
thr1 = np.arange(trace.shape[0])
plt.plot(thr1, trace)
plt.xlabel('num')
plt.ylabel('object value')
plt.title('line')


end = time.time()

print('运行时间：' , '%.2f' %((end- start)/60) , 'min')
#%%
# import seaborn as sns

# path = r'E:\科研项目\广海局-水合物-机器学习\2023年\数据\水合物形成.xlsx'
# sheetname = 'Sheet1' 
# para_o = pd.read_excel(path, str(sheetname))
# para = para_o.to_numpy()

# # -----------------R2 hotmap---------------
# name_row = para_o.columns[0:9]
# corr = np.corrcoef(para_o[name_row].T)

# plt.figure(figsize=(16,16), dpi = 300)
# sns.set(font_scale=2.0)#字符大小设定
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})

# hm=sns.heatmap(corr, vmin=-1, vmax=1, cbar=True, annot=True, square=True, fmt='.1f',
#                 xticklabels=name_row, yticklabels=name_row,cmap="YlGnBu" )
# plt.show()

# # -----------------pairplot---------------

# plt.figure(dpi= 300,figsize=(16,16))

# sns.set(font_scale=1.0)#字符大小设定
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})

# sns.pairplot(para_o[name_row], kind="reg", diag_kind="kde") 
# plt.show()

#%%

# import warnings
# warnings.filterwarnings("ignore")
# import os

# import time
# tis1 =time.perf_counter()
# import pandas as pd
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# #import tensorflow as tf#tensorflow1.x环境就用这个

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split

# from scipy.io import savemat,loadmat
# from matplotlib.pyplot import MultipleLocator

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# class DNNNet(nn.Module):
#     def __init__(self, input_features, hidden_nodes0, hidden_nodes1, hidden_nodes2, output_class):
#         super(DNNNet,self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(input_features, hidden_nodes0),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_nodes0, hidden_nodes1),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_nodes1, hidden_nodes2),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_nodes2, output_class),
#             # nn.LeakyReLU(),
#             )
#         # self.layer1 = nn.Linear(input_features, hidden_nodes0)
#         # # self.act1 = nn.LeakyReLU()
#         # self.layer2 = nn.Linear(hidden_nodes0, hidden_nodes)
#         # # self.act1 = nn.LeakyReLU()
#         # self.layer3 = nn.Linear(hidden_nodes, output_class)
#         # self.layer4 = nn.Linear(input_features, hidden_nodes0),
        


#     def forward(self, x):
#         # x = x #.view(-1,2) # Flattern the (n,1,28,28) to (n,784)
#         # x = F.leaky_relu(self.layer1(x))
#         # x = F.leaky_relu(self.layer2(x))
#         # x = self.layer3(x)
#         x = self.main(x)

#         return x

# class loaddataset():
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self,):
#         return len(self.x)
    
#     def __getitem__(self, i):
#         X = self.x[i]
#         Y = self.y[i]
#         return X, Y
    

# path = r'E:\科研项目\广海局-水合物-机器学习\论文资料\数据\1 第4组.xlsx'
# sheetname = 'Sheet2-3' 
# num =  2419
# data_frame_train = np.array(pd.read_excel(path, str(sheetname)))
    
# # data_frame = data_frame[:,:3]
# # 标准化数据
# scaler = StandardScaler().fit(data_frame_train)
# data_frame_train = scaler.transform(data_frame_train)
#  # int(len(data_frame) * 0.8)
# data = data_frame_train[:num]
# x_train = data[:, :-1]
# y_train = data[:, -1]

# path = r'E:\科研项目\广海局-水合物-机器学习\论文资料\数据\4 第1组.xlsx'
# sheetname = 'Sheet2' 

# data_frame_test = pd.read_excel(path, str(sheetname))
# str2 = data_frame_test.columns
# data_frame_test = np.array(data_frame_test)

# data = data_frame_test
# x_test = data[:, : -1]
# y_test = data[:, -1]

# alpha = 7.8410e-4# 学习率
# num_epochs = 93#迭代次数
# batch_size = 21# batchsize
# hidden_nodes0 = 39#第一隐含层神经元
# hidden_nodes1 = 69#第二隐含层神经元
# hidden_nodes2 = 37#第二隐含层神经元

# input_features = x_train.shape[1]
# output_class =  1


# model_DNN = DNNNet(input_features, hidden_nodes0, hidden_nodes1, hidden_nodes2, output_class).to('cuda')
# # print(model_DNN)
# criterion = nn.MSELoss(reduction='mean')
# optimizer = optim.Adam(model_DNN.parameters(),lr= alpha)
# train_dataset = loaddataset(x_train,y_train)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                        batch_size=batch_size,
#                                        shuffle=True)

# model_DNN.train()
# for i in range(num_epochs):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data = torch.as_tensor(data, dtype = torch.float).to('cuda')
#         target = torch.as_tensor(target, dtype =torch.float).to('cuda')
#         optimizer.zero_grad()
#         output = model_DNN(data)
#         loss = criterion(output,target)
#         loss.backward()
#         optimizer.step()
        
# model_DNN.eval()
# x = torch.as_tensor(x_test, dtype = torch.float).to('cuda')
# y = torch.as_tensor(y_test, dtype = torch.float).to('cuda')     
# with torch.no_grad():                   # 取消梯度计算(加快运行速度)
#     testPredict = model_DNN(x)                     # 前向计算
    
# realdata = np.concatenate((x_test, y_test.reshape(-1,1)),axis = 1) #  data_frame[num:, 3:-1],
# realdata = scaler.inverse_transform(realdata)
# testPredict = testPredict.detach().cpu().numpy().reshape(-1,1)
# predata = np.concatenate((x_test, testPredict),axis = 1) # data_frame[num:, 3:-1], 
# predata = scaler.inverse_transform(predata)
# # #%%
# fig = plt.figure(dpi = 300)
# plt.scatter(realdata[:,0],realdata[:,-1], label ='Origin')
# plt.plot(predata[:,0],predata[:,-1], label ='DNN')
# plt.xlabel('Time')
# plt.ylabel('K')
# plt.legend(loc='upper right')
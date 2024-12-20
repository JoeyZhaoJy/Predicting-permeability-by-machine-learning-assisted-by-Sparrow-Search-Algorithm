# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:05:34 2023

@author: Joey
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 21:03:43 2023

Ref:https://github.com/changliang5811/SSA_python
Ref:https://www.tandfonline.com/doi/full/10.1080/21642583.2019.1708830
Ref:A novel swarm intelligence optimization approach: sparrow search algorithm.pdf

@author: HP
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

import time
start = time.time()

#%% SSA

def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]-1
    return temp

# def Bounds(pop,lb,ub):
#     # 防止粒子跳出范围,除学习率之外 其他的都是整数
#     pop=[int(pop[i]) if i>0 else pop[i] for i in range(len(lb))]
#     for i in range(len(lb)):
#         if pop[i]>ub[i] or pop[i]<lb[i]:
#             if i==0:
#                 pop[i] = (ub[i]-lb[i])*np.random.rand()+lb[i]
#             else:
#                 pop[i] = np.random.randint(lb[i],ub[i])
#     return pop


def SSA(pop, M, c, d, dim, fun):
    """
    :param fun: 适应度函数
    :param pop: 种群数量
    :param M: 迭代次数
    :param c: 迭代范围下界
    :param d: 迭代范围上界
    :param dim: 优化参数的个数
    :return: 适应度值最小的值 对应得位置
    """
    P_percent = 0.2
    pNum = round(pop*P_percent)
    lb = c*np.ones((1, dim))
    ub = d*np.ones((1, dim))
    X = np.zeros((pop, dim))
    fit = np.zeros((pop, 1))

    # for i in range(pop):
    #     X[i, :] = lb+(ub-lb)*np.random.rand(1, dim)
    #     fit[i, 0] = fun(X[i, :])
    
    for i in range(pop):
        X[i, :] = np.random.randint(lb,ub)
        fit[i, 0] = fun(X[i, :])

        
    pFit = fit
    pX = X
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]
    Convergence_curve = np.zeros((1, M))
    for t in range(M):
        sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0])
        worse = X[B, :]
        # 发现者位置更新
        r2 = np.random.rand(1)
        if r2 < 0.8:
            for i in range(pNum):
                r1=np.random.rand(1)
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :]*np.exp(-(i)/(r1*M))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        elif r2 >= 0.8:
            for i in range(pNum):
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :]+np.random.rand(1)*np.ones((1, dim))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :]
        # 加入者位置更新
        for ii in range(pop-pNum):
            i = ii+pNum
            A = np.floor(np.random.rand(1, dim)*2)*2-1
            if i > pop/2:
                X[sortIndex[0, i], :] = np.random.rand(1)*np.exp(worse-pX[sortIndex[0, i], :]/np.square(i))
            else:
                X[sortIndex[0, i], :] = bestXX+np.dot(np.abs(pX[sortIndex[0, i], :]-bestXX), 1/(A.T*np.dot(A, A.T)))*np.ones((1, dim))
            X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        arrc = np.arange(len(sortIndex[0, :]))
        # c=np.random.shuffle(arrc)
        # 意识到危险得麻雀位置更新
        c = np.random.permutation(arrc)
        b = sortIndex[0, c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:
                X[sortIndex[0, b[j]], :] = bestX+np.random.rand(1, dim)*np.abs(pX[sortIndex[0, b[j]], :]-bestX)
            else:
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :]+(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0, b[j]], :]-worse)/(pFit[sortIndex[0, b[j]]]-fmax+10**(-50))
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)
            fit[sortIndex[0, b[j]], 0] = fun(X[sortIndex[0, b[j]]])
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]
        Convergence_curve[0, t] = fMin
    return fMin, bestX, Convergence_curve


#%%
def import_data():
    """
    :return: 数据导入
    """
    path = r'G:\文章成果\期刊论文\水合物储层渗透率\数据\1 第4组.xlsx'
    sheetname = 'Sheet2' 
    num =  2419
    data_frame = np.array(pd.read_excel(path, str(sheetname)))
    data = data_frame[:num]
    # 标准化数据
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    feature = data[:, : 3]
    target = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state = 1)

    return x_train, x_test, y_train, y_test


def fitness_spaction(parameter):
    """
    :param parameter: SVM参数
    :return: 最小化错误率
    """
    data_train, data_test, label_train, label_test = import_data()
    # SVM参数
    n_estimators = int(parameter[0])
    max_features = int(parameter[1])
    # bootstrap = parameter[2]
    
    forest_reg = RandomForestRegressor(n_estimators=n_estimators,max_features=max_features,
                                       bootstrap=True,
                                       )
    # forest_reg = MultiOutputRegressor(model)

    # forest_reg = RandomForestRegressor(n_estimators=n_estimators,max_features=max_features,
    #                                    bootstrap=True,
    #                                    )
    forest_reg.fit(data_train, label_train) # .ravel()
    
    y_predict = forest_reg.predict(data_test)

    mse = mean_squared_error(label_test, y_predict)
    return mse

time_pre = time.time()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time_pre))))
if __name__ == "__main__":
    SearchAgents_no = 30  # 种群数量
    Max_iteration = 50  # 迭代次数
    dim = 2  # 优化参数的个数 第一个参数n_estimators，第二个参数max_features
    lb = [1,    1]
    ub = [100,  9]
    fMin, bestX, SSA_curve = SSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction)
    SSA_curve = SSA_curve.T
    print('n_estimators和max_features为最优值时MSE为：', fMin)
    print('最优变量n_estimators:{0}, max_features:{1}'.format(int(bestX[0]), int(bestX[1])))


time_now = time.time()
time_gap = (time_now - time_pre) / 60
print('运行时间：%d min' % time_gap)

#%%
path = r'G:\文章成果\期刊论文\水合物储层渗透率\数据\1 第4组.xlsx'
sheetname = 'Sheet2' 
num =  2419
data_frame = pd.read_excel(path, str(sheetname))
str1 = data_frame.columns
data_frame = np.array(data_frame)

# 标准化数据
scaler = StandardScaler().fit(data_frame)
data_frame = scaler.transform(data_frame)

data = data_frame[:num]
x_train = data[:, : 3]
y_train = data[:, -1]

data = data_frame[num:]
x_test = data[:, : 3]
y_test = data[:, -1]

forest_reg = RandomForestRegressor(n_estimators=int(bestX[0]),max_features=int(bestX[1]),
                                   bootstrap=True,
                                   )
# forest_reg = MultiOutputRegressor(model)

forest_reg.fit(x_train, y_train)
y_predict = forest_reg.predict(x_test)
# importance =  forest_reg.feature_importances_

realdata = np.concatenate((x_test, y_test),axis = 1) # data_frame[num:, 3:-1], 
realdata = scaler.inverse_transform(realdata)

predata = np.concatenate((x_test, y_predict),axis = 1) # data_frame[num:, 3:-1], 
predata = scaler.inverse_transform(predata)
#%%
fig = plt.figure(dpi = 300)
# plt.scatter(realdata[:,0],realdata[:,-1], label ='Origin')
plt.scatter(predata[:,9],realdata[:,9], label ='Kx')
plt.scatter(predata[:,10],realdata[:,10], label ='Ky')
plt.scatter(predata[:,11],realdata[:,11], label ='Kz')
plt.xlabel('Prediction')
plt.ylabel('LBM')
plt.legend(loc='upper right')

fig = plt.figure(dpi = 300)
thr1 = np.arange(SSA_curve.shape[0])
plt.plot(thr1, SSA_curve)
plt.xlabel('num')
plt.ylabel('object value')
plt.title('line')
#%%
importanceX = forest_reg.estimators_[0].feature_importances_
ind = np.argsort(-importanceX)
str2 = str1[ind]
importanceX = importanceX[ind]
fig = plt.figure(dpi = 300)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8,4),dpi=300)
plt.bar(str2,importanceX,color='blue')
plt.xlabel('特征参数')
plt.ylabel('贡献量')
plt.xticks(rotation=50)
importanceY = forest_reg.estimators_[1].feature_importances_
ind = np.argsort(-importanceY)
str2 = str1[ind]
importanceY = importanceY[ind]
fig = plt.figure(dpi = 300)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8,4),dpi=300)
plt.bar(str2,importanceY,color='blue')
plt.xlabel('特征参数')
plt.ylabel('贡献量')
plt.xticks(rotation=50)
importanceZ = forest_reg.estimators_[2].feature_importances_
ind = np.argsort(-importanceZ)
str2 = str1[ind]
importanceZ = importanceZ[ind]
fig = plt.figure(dpi = 300)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8,4),dpi=300)
plt.bar(str2,importanceZ,color='blue')
plt.xlabel('特征参数')
plt.ylabel('贡献量')
plt.xticks(rotation=50)
# #%%
# pop = 3
# dim = 2
# X = np.zeros((pop, dim))
# lb = [1, 1]
# ub = [200,  3]
# fit = np.zeros((pop, 1))
# for i in range(pop):
#     X[i, :] = np.random.randint(lb,ub)
#     # fit[i, 0] = fun(X[i, :])

#     fit[i, 0] = fitness_spaction((X[i, :]))
# #%%
# # 预测
# import numpy as np
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# import seaborn as sns


# import pandas as pd
# # import sparrow as spa
# import matplotlib.pyplot as plt

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

# forest_reg = RandomForestRegressor(n_estimators=92,max_features=7,
#                                    bootstrap=True,
#                                    )
# forest_reg.fit(x_train, y_train.ravel())
# y_predict = forest_reg.predict(x_test)
# importance =  forest_reg.feature_importances_

# realdata = np.concatenate((x_test, y_test.reshape(-1,1)),axis = 1) #  data_frame[num:, 3:-1],
# realdata = scaler.inverse_transform(realdata)

# predata = np.concatenate((x_test, y_predict.reshape(-1,1)),axis = 1) # data_frame[num:, 3:-1], 
# predata = scaler.inverse_transform(predata)
# # #%%
# fig = plt.figure(dpi = 300)
# plt.scatter(realdata[:,0],realdata[:,-1], label ='Origin')
# plt.plot(predata[:,0],predata[:,-1], label ='RF')
# plt.xlabel('Time')
# plt.ylabel('K')
# plt.legend(loc='upper right')

# ind = np.argsort(-importance)
# str2 = str2[ind]
# importance = importance[ind]
# fig = plt.figure(dpi = 300)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# fig = plt.figure(figsize=(8,4),dpi=300)
# plt.bar(str2,importance,color='blue')
# plt.xlabel('特征参数')
# plt.ylabel('贡献量')
# plt.xticks(rotation=50)
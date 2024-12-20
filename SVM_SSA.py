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
from sklearn.svm import SVC,SVR
import pandas as pd
# import sparrow as spa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multioutput import MultiOutputRegressor
import time
#%% SSA

def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp


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

    for i in range(pop):
        X[i, :] = lb+(ub-lb)*np.random.rand(1, dim)
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
    c = parameter[0]
    g = parameter[1]
    # 训练测试
    clf = SVR(gamma=g, C=c, kernel='rbf') # ‘linear’， ‘poly’， ‘rbf’， ‘sigmoid’，‘precomputer’
    # clf = MultiOutputRegressor(model)
    
    clf.fit(data_train, label_train)
    y_predict = clf.predict(data_test)
    # acc = accuracy_score(label_test, y_predict)
    # return 1 - acc
    mse = mean_squared_error(label_test, y_predict)
    return mse
#%%
time_pre = time.time()
'''
确定最优参数
'''
path = r'G:\文章成果\期刊论文\水合物储层渗透率\数据\1 第4组.xlsx'
sheetname = 'Sheet2' 
num =  2419
if __name__ == "__main__":
    SearchAgents_no = 30  # 种群数量
    Max_iteration = 50  # 迭代次数
    dim = 2  # 优化参数的个数
    lb = [0.001, 0.001]
    ub = [100,    100]
    
    # path = r'E:\科研项目\广海局-水合物-机器学习\2023年\数据\通量.xlsx'
    # sheetname = '2020CO2' 
    data_frame = np.array(pd.read_excel(path, str(sheetname)))
    # num = 2422 # int(len(data_frame) * 0.8)
    
    fMin, bestX, SSA_curve = SSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction)
    SSA_curve = SSA_curve.T

    print('c和g为最优值时MSE为：', fMin)
    print('最优变量c:{0}, g:{1}'.format(bestX[0], bestX[1]))

time_now = time.time()
time_gap = (time_now - time_pre) / 60
print('运行时间：%d min' % time_gap)

#%%

# 预测

data_frame = np.array(pd.read_excel(path, str(sheetname)))
# data_frame = data_frame[:,:3]
# 标准化数据
scaler = StandardScaler().fit(data_frame)
data_frame = scaler.transform(data_frame)
 # int(len(data_frame) * 0.8)
data = data_frame[:num]
x_train = data[:, :3]
y_train = data[:, -1]

data = data_frame[num:]
x_test = data[:, : 3]
y_test = data[:, -1]

clf_best = SVR(gamma=bestX[1], C=bestX[0])
# clf_best = MultiOutputRegressor(model)
clf_best.fit(x_train, y_train)
y_predict = clf_best.predict(x_test)

realdata = np.concatenate((x_test, y_test),axis = 1) #  data_frame[num:, 3:-1],
realdata = scaler.inverse_transform(realdata)

predata = np.concatenate((x_test, y_predict),axis = 1) # data_frame[num:, 3:-1], 
predata = scaler.inverse_transform(predata)
# #%%
fig = plt.figure(dpi = 300)
# plt.scatter(realdata[:,0],realdata[:,-1], label ='Origin')
plt.scatter(predata[:,9],realdata[:,9], label ='Kx')
plt.scatter(predata[:,10],realdata[:,10], label ='Ky')
plt.scatter(predata[:,11],realdata[:,11], label ='Kz')
plt.xlabel('Prediction')
plt.ylabel('LBM')
plt.legend(loc='upper right')

fig = plt.figure(dpi = 300)
thr1 = np.arange((SSA_curve.shape[0]))
plt.plot(thr1, SSA_curve)
plt.xlabel('num')
plt.ylabel('fittness value')
plt.title('line')

# #%% 预测
# path = r'E:\科研项目\广海局-水合物-机器学习\2023年\结题\地层厚度.xlsx'
# sheetname = '冻土层厚度' 
# para_o = pd.read_excel(path, str(sheetname))
# para = para_o.to_numpy()

# #-----------------R2 hotmap---------------
# name_row = para_o.columns
# corr = np.corrcoef(para_o[name_row].T)
# plt.rcParams['axes.unicode_minus']=False 
# plt.figure(figsize=(16,16), dpi = 300)
# sns.set(font_scale=2.0)#字符大小设定
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})

# hm=sns.heatmap(corr, vmin=-1, vmax=1, cbar=True, annot=True, square=True, fmt='.1f',
#                 xticklabels=name_row, yticklabels=name_row,cmap="YlGnBu" )
# plt.show()

# # -----------------pairplot---------------
# plt.rcParams['axes.unicode_minus']=False 
# plt.figure(dpi= 300,figsize=(16,16))

# sns.set(font_scale=2.0)#字符大小设定
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})

# sns.pairplot(para_o[name_row], kind="reg", diag_kind="kde") 
# plt.show()

#%%
# 预测
# import numpy as np
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC,SVR
# import pandas as pd
# # import sparrow as spa
# import matplotlib.pyplot as plt
# import seaborn as sns

# path = r'E:\科研项目\水合物-机器学习\论文资料\数据\静态孔隙结构参数.xlsx'
# sheetname = 'Sheet6' 
# num =  1000
# data_frame_train = np.array(pd.read_excel(path, str(sheetname)))
    
# # data_frame = data_frame[:,:3]
# # 标准化数据
# scaler = StandardScaler().fit(data_frame_train)
# data_frame_train = scaler.transform(data_frame_train)
#  # int(len(data_frame) * 0.8)
# data = data_frame_train[:num]
# x_train = data[:, :-3]
# y_train = data[:, -3:]

# path = r'E:\科研项目\水合物-机器学习\论文资料\数据\静态孔隙结构参数.xlsx'
# sheetname = 'Sheet6' 
# data_frame_test = np.array(pd.read_excel(path, str(sheetname)))
# data = data_frame_test
# x_test = data[:, : -3]
# y_test = data[:, -3:]

# # clf_best = SVR(gamma=0.6223, C=1.0906)
# # model = SVR(gamma=bestX[0], C=bestX[1])
# # clf_best = MultiOutputRegressor(model)
# clf_best.fit(x_train, y_train)
# y_predict = clf_best.predict(x_test)

# realdata = np.concatenate((x_test, y_test.reshape(-1,1)),axis = 1) #  data_frame[num:, 3:-1],
# realdata = scaler.inverse_transform(realdata)

# predata = np.concatenate((x_test, y_predict.reshape(-1,1)),axis = 1) # data_frame[num:, 3:-1], 
# predata = scaler.inverse_transform(predata)
# # #%%
# fig = plt.figure(dpi = 300)
# plt.scatter(realdata[:,0],realdata[:,-1], label ='Origin')
# plt.plot(predata[:,0],predata[:,-1], label ='SVR')
# plt.xlabel('Time')
# plt.ylabel('K')
# plt.legend(loc='upper right')


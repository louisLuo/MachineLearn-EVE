# 我们对数据的存储使用 pandas
import pandas as pd
 # 数据的分析和使用，我这里使用 numpy
import numpy as np
#我们对数据的展示，使用 pyplot
import matplotlib.pyplot as plt

import scipy.io as sio

#引入算法
import linearRegression


# ==========================
# linear regression

# 获取外部数据与转换
data = pd.read_csv('Data/ex1data1.CSV')
data = data.as_matrix()  # 将数据转化为矩阵
X = np.array(data[:,0]) #创建 np 的一个数组哦
y = np.array(data[:,1])

# 开始初始化 参数和参数尺寸
m = len(y)
theta = np.zeros((2, 1))
theta = np.matrix('-1 ; 2')

# 开始初始化 hypermeter
iterations = 1500;
alpha = 0.01;

# 这个方法中 x 必须在前面全部加1
Xones = np.column_stack((np.ones((m,1)),X))
thetaP,Jhistory = linearRegression.linearRegression(Xones,y,theta,alpha,iterations,False)

#测试所有alpha的值
alpha = np.array([0.3, 0.1, 0.03, 0.01])
linearRegression.alphaRateTest(Xones,y,theta,alpha,iterations)

# 绘制前我需要将函数的 prodect 结果放进去哦
plt.plot(X,y,'r+')  # 绘制点
plt.plot(X,np.dot(Xones,thetaP),'-',LineWidth=1)
plt.show()

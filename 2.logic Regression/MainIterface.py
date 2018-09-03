# 我们对数据的存储使用 pandas
import pandas as pd
 # 数据的分析和使用，我这里使用 numpy
import numpy as np
#我们对数据的展示，使用 pyplot
import matplotlib.pyplot as plt

import scipy.io as sio

#引入算法
import logisticRegression


# ==========================
dataMat = sio.loadmat('Data/ex3data1.mat')
# print(dataMat['X'])
X = np.mat(dataMat['X'])
y = np.mat(dataMat['y'])
# 将数据打乱，然后取 6 2 2 比例那train 数据
sourceAll = np.column_stack((X,y))
p = np.random.permutation(sourceAll.shape[0])
X = sourceAll[p, :][:,0:-1]
y = sourceAll[p, :][:,-1]

# trainNumber = len(X[:,0])*.6
m,n = X.shape
trainNumber = int(m*.6)
CVNumber = int(trainNumber+m*.2)
testNumber = int(CVNumber+m*.2)

classssIndex = np.where(y==10)[0] # 找到所有的 10 把他们全部变成0
y[classssIndex,:] = 0

Xtrain = X[0:trainNumber,:]
ytrain = y[0:trainNumber,:]

XCV = X[trainNumber:CVNumber,:]
yCV = y[trainNumber:CVNumber,:]

# Xtrain = X
Xtest = X[trainNumber:testNumber,:]
ytest = y[trainNumber:testNumber,:]

pictureWidth = 20
pictureHight = 20

# 一群数据拿出显示
display_rows = 10
display_cols  = 10
index = 0
# 数据是有方向的，这里的照片的数据方向也是不同的，所以需要添加 .T 这个方法来让数据变得正常
for i in range(0,display_cols):
    for j in range(0,display_rows):
        gridData = Xtrain[index,:]
        gridData = gridData.reshape((pictureWidth,pictureHight))
        try:
            gridRow = np.row_stack((gridRow,gridData.T))
        except Exception as e:
            gridRow = gridData.reshape((pictureWidth,pictureHight)).T
        index = index+1;
    try:
        grid = np.column_stack((grid,gridRow))
    except Exception as e:
        grid = gridRow
    gridRow = [] #每次添加好之后，需要立刻清理缓存数据
    # print(grid.shape)


# 绘制图像的案例
# fig, (ax2, ax3) = plt.subplots(nrows=2, figsize=(6,10)) #figsize 是比例
# ax1.imshow(grid, extent=[0,100,0,1])
# ax1.set_title('Default')

# ax2.imshow(grid0, extent=[0,100,0,1], aspect='auto',cmap='gray')
# ax2.set_title('Auto-scaled Aspect')

# fig, ax3 = plt.subplots()
# ax3.imshow(grid,extent = [0,100,0,100],aspect='auto',cmap='gray')
# ax3.set_title('Manually Set Aspect')
#
# plt.tight_layout()
# plt.show()

# one for all
m,n = Xtrain.shape
iterations = 1
lambdas = 1
alpha = 1
Xones = np.column_stack((np.ones((m,1)),Xtrain))

m,n=Xones.shape
theta = np.random.random_sample((1,n))
# theta = np.zeros((1,n))
# print(theta.shape,"theta")
# thetaP = logisticRegression.oneVsAll(Xones[0:100,:],ytrain[0:100,:],10,ifMap=False)
thetaP = logisticRegression.oneVsAll(Xones,ytrain,10,ifMap=False)
# print(thetaP)

m,n = Xtest.shape;
XonesTest = np.column_stack((np.ones((m,1)),Xtest))
yp = logisticRegression.oneVsAllPridiction(XonesTest,thetaP)

yss = yp - ytest
ym,yn = yss[np.where(yss==0)[0],0].shape
print(ym/len(ytest))

# 随机获取一个张图，然后识别他
# fig, ax3 = plt.subplots()
# ax3.imshow(grid,extent = [0,100,0,100],aspect='auto',cmap='gray')
# ax3.set_title('Manually Set Aspect')
#
# plt.tight_layout()
# plt.show()

pictureWidth = 20
pictureHight = 20
currentPicIndex = 0
while True:
    orders = input("Type any words: (type quit to quit)")
    if(orders=="quit"):
        break
    Xpicture = Xtest[currentPicIndex,:]

    m,n = Xpicture.shape;
    Xpicture = np.column_stack((np.ones((m,1)),Xpicture))
    yp = logisticRegression.oneVsAllPridiction(Xpicture,thetaP)
    print(yp)


    # 取一张照片
    gridData0 = Xpicture[:,1::]
    # 重新排列成图片的样子
    grid0 = gridData0.reshape(pictureWidth,pictureHight).T
    fig, ax3 = plt.subplots()
    ax3.imshow(grid0,extent = [0,10,0,10],aspect='auto',cmap='gray')
    ax3.set_title(' we get a picture')
    plt.tight_layout()
    plt.show()

    currentPicIndex=currentPicIndex+1

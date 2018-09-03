# supervised learning
#linear Regression


# step1 :   feature normalize
# step2 : 	gradient descent
# step3 :	alpha rate

# step4 : 我们有了一组比价可靠的 alhpa 和 iteration 值
# step5 : 大量大量的进行数据计算




import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

def alphaRateTest(X, y, theta,alpha,num_iters):
    J_history_rate = np.zeros((1,len(alpha)))
    linearS = np.linspace(0,len(alpha),len(alpha),endpoint=False)
    # for alphaRate = 1:size(alpha,1)
    for alphaRate in range(0,len(alpha)):
        thetaP,J_history_rate[0,alphaRate] = linearRegression(X, y, theta,alpha[alphaRate] , num_iters,False)

    plt.plot(alpha.reshape(-1,1),J_history_rate.reshape(-1,1),'go--',LineWidth = 1)  # 绘制 + 号
    plt.show()



#归化所有的数据
# 由于输入的数据带有ones 所以要剥离前面的ones
# 通过归化，将大数据比如超过万，几万，几十万的大型数据变成小型的数据，以免这个超大量的数据影响到数据的真实性
# 其实你在预测的时候，依然走的也是数据 规范化的计算方式，所以，进来之后一样也是归化过的数据进行计算的，所以在结果上面没有区别。
def featureNormalize(X):
    X = X[0::,1::]; #对X 进行矩阵截取 操作，这里和 matlab不同，前面一个代表的是行数，逗号后面代表的是列数
    m,n = X.shape;
    mu = np.zeros((1,n));
    sigma = np.zeros((1,n));
    X_normal = X;
    for i in range(0,n):
        mu[0,i] = np.mean(X[:,i]);
        sigma[0,i] = np.std(X[:,i]);
        X_normal[:,i] = (X[:,i] - mu[0,i]) / sigma[0,i];

    #做完后需要在前面加ones
    X_normal = np.column_stack((np.ones((m,1)),X_normal));
    return X_normal

def computeCost(X,y,theta):
    #格式化所有数据 y 必须是列的
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1,1);
    m = len(y)
    theta = np.array(theta)
    Hx = np.dot(X,theta)
    J = (np.power((Hx - y),2).sum())/(2*m)
    return J


#绘制曲线，我们看这个方程式是不是在完美的运行中
def mapDraw(X,y,theta,theta_history,J_history):
    num_iters = len(J_history)
    #绘制J 曲线图 ，看看J 的曲线是否正常
    linearS = np.linspace(0,num_iters,num_iters,endpoint=False) #0 起始位置 num_iters 终止位置大小，num_iters 中间的跳转次数
    plt.plot(linearS,J_history,'--',LineWidth=1)
    plt.xlim(0, num_iters)  # 设定边界范围
    plt.show()

    # #绘制 梯降 3d图
    # #创建模版
    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # theta0_vals = np.linspace(-10, 2.5, 100);
    # theta1_vals = np.linspace(0, 3, 100);
    # J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));
    #
    # # % Fill out J_vals
    # for i in range(0, len(theta0_vals)):
    #     for j in range(0, len(theta1_vals)):
    #         print(i,j)
    #         t = [theta0_vals[i],theta1_vals[j]]
    #         J_vals[i,j] = computeCost(X, y, t);
    #
    # #格式化数据 适应 3D
    # # theta0_data = theta0_vals*np.ones((len(theta1_vals),1)).T;
    # # theta1_data = theta1_vals*np.ones((len(theta0_vals),1));
    # # theta0_data = theta0_data.reshape(-1,);
    # # theta1_data = theta1_data.reshape(-1,);
    # # J_vals = J_vals.reshape(-1,);
    # # print(theta0_data.shape,theta1_data.shape,J_vals.shape)
    # # ax.plot_trisurf(theta0_data, theta1_data, J_vals) #绘制点 x y 都必须是1维度的
    #
    # # plot_surface 的方式
    # theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    # # print(theta0_vals) #纵向全部一样
    # # print(theta1_vals) #横向全部一样
    # ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    #
    #
    # # 其他用法
    # # ax.scatter(theta_history[0,:].reshape(-1,), theta_history[1,:].reshape(-1,), J_history.reshape(-1,)) #绘制曲线
    # # ax.plot(theta_history[0,:].reshape(-1,), theta_history[1,:].reshape(-1,), J_history.reshape(-1,)) #绘制曲线
    # # ax.plot_trisurf(theta_history[0,:].reshape(-1,), theta_history[1,:].reshape(-1,), J_history.reshape(-1,)) #绘制点图
    # ax.scatter(theta_history[0,:].reshape(-1,), theta_history[1,:].reshape(-1,), J_history.reshape(-1,), c = 'r', marker = '^') #点为红色三角形
    # ax.scatter(theta[0],theta[1],computeCost(X, y, theta), c = 'r', marker = 'x',)
    # plt.show()

 #主要的运行函数
 # 总结，
 # 1在使用函数计算矩阵后，注意样将他们重新定义为为 np.array 不然你在计算的时候，矩阵很有可能在方向上错误
 # 2在矩阵计算的时候，永远要测试矩阵的方向
 #
 # 函数输入
 # X 必须带 X0 的 m*n 的函数
 # y 必须是 m*1 的函数
 # theta 必须是 n*1 的函数
 # alpha 一般用 0.01
 # num_iters 次数我用了2000 ，差不多的样子
 # ifMap 是否打开图片显示
def linearRegression(X,y,theta,alpha,num_iters,ifMap):
    #格式化数据
    # number of training examples
    m = len(y);
    # all history store here
    J_history = np.zeros((num_iters, 1));
    theta_history = np.zeros((2, num_iters));
    J = 0;
    #格式化所有数据 y 必须是列的
    y = y.reshape(-1,1);
    #保留原数据格式 格式化 theta 必须是一列数据
    theta = theta.reshape(-1,1);
    # 归化所有样本数据
    X = featureNormalize(X);

    #这里的算法以后需要修改
    iteration = 0
    while iteration<num_iters:
        print("===========",iteration)
        J_history[iteration] = computeCost(X, y, theta);
        theta_history[:,iteration] = theta.reshape(2,); #如果格式不正确，那就把格式统一一下咯，就good了

        Hx = np.array(np.dot(X,theta)) #他的精确值没有问题 注意注意这里的dot 要想正常运算都需要np.array 一下哦

        result = np.array(Hx-y); #为什么要转一下，因为你所有的* 都必须是 array 不能是matrix

        #vertical sum!!!
        theta = theta.T - alpha*np.sum(result*X,axis=0)/m;
        theta = theta.T;

        #back the J
        J = J_history[iteration];

        #final plus one
        iteration = iteration+1;

    if (ifMap == True):
        mapDraw(X,y,theta,theta_history,J_history)
    return [theta,J]

import numpy as np
import matplotlib.pyplot as plt

# step1 :   Data change 我们将元数据分成 train test cv
# step1 :   one for all 进行分类，注意这里的 y 不是 类目，而是不同类目的值，为0，这个类目的值改成1 x 不变进行训练
# step2 : 	gradient descent 自动化，如果下一个值为负数，就将alpha小的下一个级别，迭代1000次
# step3 :   predict 预测的话，我们就是需要把 sigam 拿出来计算哪一个类目最接近1，那么预测就是他啦。

# step4 : 我们有了一组比价可靠的 alhpa 和 iteration 值
# step5 : 大量大量的进行数据计算




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

def machineLearningDataNormalize(x):
    x = x[0::,1::]; #对X 进行矩阵截取 操作，这里和 matlab不同，前面一个代表的是行数，逗号后面代表的是列数
    m,n = x.shape;
    x_norm = np.linalg.norm(x,axis = 1,keepdims = True)
    x = x/x_norm
    x = np.column_stack((np.ones((m,1)),x));
    return x

# 测试每一个 alpha 值对于模型的最优化结果 如果时nan，那么这个alpha就不可用
def alphaRateTest(X, y, theta,alpha,num_iters):
    J_history_rate = np.zeros((1,len(alpha)))
    linearS = np.linspace(0,len(alpha),len(alpha),endpoint=False)
    # for alphaRate = 1:size(alpha,1)
    for alphaRate in range(0,len(alpha)):
        J_history_rate[0,alphaRate],thetaP = logisticRegression(X, y, theta,alpha[alphaRate] , num_iters,False)

    plt.plot(alpha.reshape(-1,1),J_history_rate.reshape(-1,1),'go--',LineWidth = 1)  # 绘制 + 号
    plt.show()

# 这个算法的核心思想，就是每一层的 degree 都加一遍，加到目标 degree
def featureMapPlus(X,degree):
    X = X[0::,1::];
    limitDegree = degree + 1;
    m,number4Feature = X.shape;
    outX = np.ones((m,1))

    #$$$$$$$ 绘制一个全MAP用来计算
    #由于数据没法初始化为一个空值，所以先把 第一个值拿出来，所以degree从1开始算
    #part 1
    totalListMap = (np.linspace(0,limitDegree,limitDegree,endpoint=False).reshape(-1,1));#这是第一级
    totalLen = len(totalListMap[:,0]);
    degreeMap = np.zeros((totalLen,1));

    # part 2 and than
    for xIndex in range(1,number4Feature):
        # print("number4Feature==========",xIndex)
        # degree 个 copy 的 feature 矩阵  +  totalListMap 长度的 0 - degree 的矩阵。
        featrueSource = totalListMap;
        totalLen = len(totalListMap[:,0]);
        for deepIndex in range(1,limitDegree):
            totalListMap = np.row_stack((featrueSource.copy(),totalListMap));
            degreeMap = np.row_stack((deepIndex*np.ones((totalLen,1)),degreeMap));
        totalListMap = np.column_stack((totalListMap,degreeMap));

    #$$$$$$$ 有了map 然后获取所有 和等于 degree的值
    # &&条件 ， 最大值不能超过 degree 的行
    # &&条件， 总和 为 degree 的行
    # 设定参数
    maplistLen = len(totalListMap[:,0]);
    # print(totalListMap)
    # 添加新X
    for degreeLayer in range(1,limitDegree):
        # print("degreeLayer==========",degreeLayer)
        # targetMap = np.where(totalListMap>degreeLayer)
        # print(maplistLen)
        for index in range(0,maplistLen):
            tempMatrix = totalListMap[index,:]
            # print(np.max(tempMatrix)>degreeLayer)
            if(np.max(tempMatrix)>degreeLayer):
                continue;
            if(np.sum(tempMatrix)==degreeLayer):
                # print(tempMatrix)
                rightX = 1;
                for xIndex in range(0,number4Feature):
                    rightX = rightX*(X[:,xIndex]**tempMatrix[xIndex]);
                outX = np.column_stack((outX,rightX));

    # print(outX.shape)
    theta = np.zeros((len(outX[0,:]),1));
    return [outX,theta]


# 逻辑分析模型的s 型算法
def sigmoid(z):
    # gx = (1+np.exp(z*-1))**-1;
    gx = 1+np.exp(z*-1);
    gx = np.power(gx,-1); #也可以这么写
    return gx

# 没有经过 正规化的算法
def costFunction(theta, X, y):
    #格式化所有数据
    theta = theta.reshape(-1,1);
    y = y.reshape(-1,1);

    z = np.dot(X,theta)
    m = len(y)
    h = sigmoid(z)
    J =  np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m;
    thetaGrad = np.sum((h-y)*X,0)/m;

    #输出时也需要重新将对象的格式规定一下
    thetaGrad = thetaGrad.reshape(-1,1);
    return [J,thetaGrad]

#   经过 正规化的算法
#   Regularized logistic regression
def costFunctionReg(theta, X, y,lambdas):
    #格式化所有数据
    theta = theta.reshape(-1,1);
    y = y.reshape(-1,1);
    m = len(y)
    #计算数据
    z = np.dot(X,theta)
    h = sigmoid(z)
    # x0 不能算进去 所以 theta 0 也不能算进去
    TheataJ = theta[1::,:];
    J =  np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m + lambdas*np.sum(TheataJ**2)/2/m;
    X0 = X[:,0].reshape(-1,1);
    grad0 = np.sum((h-y)*X0)/m;
    gradj = ((np.sum((h-y)*X[:,1::],0)).T.reshape(-1,1) + TheataJ*lambdas)/m;
    thetaGrad = np.row_stack((grad0,gradj));

    #输出时也需要重新将对象的格式规定一下
    thetaGrad = thetaGrad.reshape(-1,1);
    return [J,thetaGrad]

#显示所有 J 的曲线，来展示，我们现在的算法是否已经到达了极限
def showJHistoryMap(J_history):
    m,n = J_history.shape
    print(J_history.shape,"J_history.shape")
    iteration = np.linspace(0,m,m)
    plt.plot(iteration,J_history,'-',linewidth=1)
    plt.show()

# 逻辑回归的主入口
def logisticRegression(X,y,theta,alpha=1,num_iters=1,lambdas=1,ifMap=False,ifFeatureMap=False,ifFeatureNormalize=False):
    X = np.array(X)
    y = np.array(y)
    #添加 feature
    # X,theta = featureMapPlus(X,1)
    # print(theta.shape)
    #按照惯例归化一下数据
    if(ifFeatureNormalize):
        X = featureNormalize(X)
    #格式化所有数据
    theta = theta.reshape(-1,1);
    y = y.reshape(-1,1);
    J_history = np.zeros((num_iters,1))
    #这里的算法以后需要修改
    for i in range(0, num_iters):
        # print("===================",i)
        J_history[i], thetaGrad = costFunctionReg(theta,X,y,lambdas)
        theta = theta - alpha*thetaGrad;
    if (ifMap == True):
        print(J_history)
        showJHistoryMap(J_history)
    return [J_history[num_iters-1],theta]

# 通过自动化的形式，自动的得到最优化数据
def logisticRegressionAuto(X,y,theta,ifMap):
    print("============ auto logistic regression start  ================")
    alpha = np.array([0.3, 0.1, 0.03, 0.01,0.001,0.0001,0.00005,0.00001])
    alphaTag = 0
    thetaGrad = 0
    num_iters=1000
    lambdas=10
    ifFeatureMap=False
    ifFeatureNormalize=True
    returnJ = 0;
    X = np.array(X)
    y = np.array(y)
    #添加 feature
    # X,theta = featureMapPlus(X,1)
    # print(theta.shape)
    #按照惯例归化一下数据
    if(ifFeatureNormalize):
        X = machineLearningDataNormalize(X)
    #格式化所有数据
    theta = theta.reshape(-1,1);
    y = y.reshape(-1,1);
    # J_history = np.zeros((1,1))
    #这里的算法以后需要修改
    for i in range(0, num_iters):
    # while(True):
        # print("===================",i)
        goTheta = theta - alpha[alphaTag]*thetaGrad
        J, thetaGrad = costFunctionReg(goTheta,X,y,lambdas)
        if(J>0):
            try:
                J_history = np.row_stack((J_history,J))
            except Exception as e:
                J_history = J.reshape(-1,1)
            theta = goTheta
            returnJ = J
        else:
            alphaTag = alphaTag+1;
            if alphaTag>=len(alpha):
                break
            continue
    if (ifMap == True):
        print(J_history,"J_history")
        showJHistoryMap(J_history)

    print("============ auto logistic regression Finish ================")
    return [returnJ,theta]

# 图像识别算法入口
# 注意注意 one vs all y 值不能是 class 而是0和1 啊！！！！，所以这个类目的其他X 和 Y 都能用到，Y 值永远只有0 和 1
def oneVsAll(X,y,num_labels,ifMap=False):
    m,n = X.shape
    all_theta = np.ones((num_labels, n));
    for classsNum in range(0,num_labels):
        classssIndex = np.where(y==classsNum)[0]#由于y 是一维的，所以只需要拿到 column 上面的值就可以
        tempY = np.zeros((m,1)) # 获取 生成所有 0 的y
        tempY[classssIndex,:] = 1
        theta = np.zeros((n,1))
        J,thetas = logisticRegressionAuto(X,tempY,theta,ifMap)
        all_theta[classsNum,:] = thetas.reshape((n,))
    return all_theta;

def oneVsAllPridiction(X,theta):
    py = 1;
    m,n = X.shape
    classsN,n = theta.shape
    yPredic = np.zeros((m,1))
    for Xmun in range(0,m):
        Xtest = X[Xmun,:]
        yHistory = np.ones((classsN,1))
        for classsNum in range(0,classsN):
            z = np.dot(Xtest,theta[classsNum,:])
            yHistory[classsNum,0] = sigmoid(z)
        # print(yHistory)
        yPredic[Xmun,0] = yHistory.argmax()
    return yPredic

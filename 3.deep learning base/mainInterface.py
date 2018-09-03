import os #用于文件操作处理。
# 我们对数据的存储使用 pandas
import pandas as pd
 # 数据的分析和使用，我这里使用 numpy
import numpy as np
#我们对数据的展示，使用 pyplot
import matplotlib.pyplot as plt
#引入 数据倒入处理对象
import scipy.io as sio
# 引入 h5py 数据处理对象，### 目前没有用到
import h5py
import pickle

# 中心思想：
# 1. 列出整个过程
# 2. 将重复使用的部分 工具化
# 3. 将关键数据拿出来 初始化
# 4. 将可重复使用的部分拿出来 模块化


#  \\\\\\\\\\\\\\ 区域需要优化
# ************ 区域需要手动输入


# ================== 1.hypermeter and basedata initialize ==================
# 将数据随机化，然后取 6 2 2 比例那train 数据
def initializeData(X,y):
    sourceAll = np.column_stack((X,y))
    m,n = X.shape
    ym,yn = y.shape
    p = np.random.permutation(sourceAll.shape[0])
    X = sourceAll[p, :][:,0:n]
    y = sourceAll[p, :][:,n:]

    print(X.shape)

    # 设定train，CV，Test 的data
    trainNumber = int(m*.6)
    CVNumber = int(trainNumber+m*.2)
    testNumber = int(CVNumber+m*.2)
    Xtrain = X[0:trainNumber,:]
    ytrain = y[0:trainNumber,:].T
    XCV = X[trainNumber:CVNumber,:]
    yCV = y[trainNumber:CVNumber,:].T
    Xtest = X[trainNumber:testNumber,:]
    ytest = y[trainNumber:testNumber,:].T

    return [Xtrain,ytrain,XCV,yCV,Xtest,ytest]


# print(Xtrain.shape)
# print(y.shape)

# initialize hypermeter
# 从外部获取 hyper meter 的时候，是不是格式会不一样？\\\\\\\\\\\
# 注意 矩阵的矢量和size
# 使用b 的好处，就是你计算X的时候，不需要验证 前面有没有ones这种操作。
# 架构现在为 input hidelayer output 3个部分
# deep learning 的好处之一是将本来你需要oneforall 的logic regression，变成1个模型进行连续深度的计算。

# 这里我们设计的图形里面只有10个数字对吧
# hyper meter 上面 w b 这些参数的尺寸和 X的m 无关
#有一个原则性的东西，就是这里的所有对象的值，尺寸，都必须初始化好，尤其是尺寸，矩阵尺寸在计算时不可变
# 在 全局里面，我决定使用一个 goalbol permeter  = tempdaat | hypermeter 这样子
# 我们用 hypermeter 进行存取 超级参数，从而可以还原算法的过程，我们除了初始化之外，还有一些方案，比如通过object，通过
# 遍历这个object 的名称(排序无先后)，得到所有你声明的参数值，然后从hypermeter里面拿出来存储。
# 格式上，所有的数据都不应该是一个int这种类型，而是array这种类型 ，起码有[0]
# 所有数据必须有统一的数据格式，这里使用的是数组，这会影响整个算法公式
def initializeHypermeters(output_y,Xn,input_hypermeter=None):
    if(input_hypermeter==None):
        hypermeter = dict()
        hypermeter['hidelayer'] = [200]
        # hypermeter['hidelayer'] = []

        # \\\\\\\\\\\\\\\\\ w 到底可以不可以使用多维数组 而不是 dict?
        hypermeter['w'] = []
        hypermeter['b'] = []
        hypermeter['iteration']=[1500]
        hypermeter['alpha']=[0.8]
        hypermeter['lambda'] = [10] # if this lambda is 0 , its mean you have no regularization here
        hypermeter['output'] = [output_y]
        # hypermeter['w'][0] = np.zeros((hypermeter['hidelayer'][0],n))
        # 这里的w不可以为0 所以我们进行了随机负值，让里面的值接近与0
        # 为什么b是n0，1 因为对于 a0来说，所有从 x 到 a0的公式都是一样（权重，b 都是一样的）的，所以只有n1 个 b
    else:
        obj = []
        obj.append('hidelayer')
        obj.append('w')
        obj.append('b')
        obj.append('iteration')
        obj.append('alpha')
        obj.append('lambda')
        obj.append('output')

        hypermeter = dict()
        for name in obj:
            # print(name,np.array(hypermeter[name]))
            # hypermeter[name] = input_hypermeter[name]
            hypermeter[name] = np.array(input_hypermeter[name][0])
            print(name,hypermeter[name])
            # hypermeter[name] = np.mat(input_hypermeter[name][0])

            # train_dataset = h5py.File('datasets/train_signs.h5', "r")
    # train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    # train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    #
    # test_dataset = h5py.File('datasets/test_signs.h5', "r")
    # test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    # test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    #
    # classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    #
    # train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    # test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    #
    # return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


    #------- temp data 这些数据都是本程序需要的值，和其他无关
    hypermeter['a'] = []
    hypermeter['z'] = []
    hypermeter['dZ'] = []

    #将 input layer , hidelayer,outputlayer 组合在一起进行计算
    totalLayer = [n] #input
    if(len(hypermeter['hidelayer'])>0):
        for i in hypermeter['hidelayer']:
            totalLayer.append(i)
    totalLayer.append(hypermeter['output'][0])
    hypermeter['totalLayer'] = totalLayer

    # print(totalLayer[1])
    #因为 这里中间有数值的计算层比 总共的数据层 少 1层。所以要剪掉1.
    if len(hypermeter['w'])<1:
        for i in range(0,len(totalLayer)-1):
            print('current layer ::::',i)
            # 注意 W 是进行过 T 的结果，我就不运算 T 方法，直接就给初始化了
            # 这里使用了 防止 vanish 和 explode 技术
            hypermeter['w'].append(np.random.randn(totalLayer[i+1],totalLayer[i])*np.sqrt(2./totalLayer[i]))
            hypermeter['b'].append(np.zeros((totalLayer[i+1],1)))
        hypermeter['w'] = np.array(hypermeter['w'])
        hypermeter['b'] = np.array(hypermeter['b'])


    # 下面的对象怎么初始化都没有问题，用来计算的时候进行缓存只要尺寸对就行
    # \\\\\ 这里是不是涉及到 深度存储的问题？
    # hypermeter['dW'] = np.array(hypermeter['w'])*0
    hypermeter['dW'] = np.array(hypermeter['w'])
    hypermeter['dB'] = np.array(hypermeter['b'])
    # hypermeter['a']  = np.array(hypermeter['w'])# A 很特别，因为他有0 其他都没有0 ，都从1开始，这里所有的0都表示数学里面的1，所以不能用0，而应该用X表示
    # hypermeter['z']  = np.array(hypermeter['a'])
    # hypermeter['dZ'] = np.array(hypermeter['b'])
    return hypermeter


# ================== 2.check fp function green ==================
# 根据模型上的数学公式进行转换代码
# \\\\\\\\\\\\\\ 正则化？
def FowradPropagationFC(hypermeter,X,y):
    activation = X.T
    hypermeter['a'] = []
    hypermeter['z'] = []
    hypermeter['dZ'] = []
    # loop
    # z = W*A + B
    # a = g(z) = sigmod(Z)

    #因为是FP所以从小到大
    for i in range(0,len(hypermeter['w'])):
        z = np.dot(hypermeter['w'][i],activation) + hypermeter['b'][i]
        # activation = 1.0 ./ (1.0 + exp(-z));
        activation = np.power(1+np.exp(z*-1),-1); #也可以这么写
        hypermeter['a'].append(activation)
        hypermeter['z'].append(z)
        hypermeter['dZ'].append(z)

    print('System ::: FP calculate finished')
    print('================')

    return activation


# ================== 3.check bp function green ==================
# 根据模型上的数学公式进行转换代码
# 算 L = Zj
#loop 这个公式的关键就是对Z进行求导计算
# 算 0
# g(z)' = g(z)*(1-g(z))
# \\\\\\\\\\\\\\ 正则化？
def backPropagationFC(hypermeter,X,y):
    m,n = X.shape
    #启动数学式子
    # regularization = np.linalg.norm(hypermeter['w'][len(hypermeter['dZ'])-1])*1/2/m
    # regularizationMy = np.power(np.sum(np.sum(np.array(hypermeter['w'][len(hypermeter['dZ'])-1])**2)),0.5)*hypermeter['lambda']/2/m
    #因为公式上是用的 norm 的 平方来进行计算的。
    # 每一层都是一层 比如逻辑回归的演算，所以，其实每一层的w 是不相关的，你只需要知道，每一层的结果output就是下一层的开始。

    # print('regularization',regularization,regularizationMy)
    # print(y.shape)
    hypermeter['dZ'][len(hypermeter['dZ'])-1] = hypermeter['a'][len(hypermeter['a'])-1] - y;
    # print(hypermeter['dZ'][len(hypermeter['dZ'])-1].shape)
    #因为 0 是特例，所以与其用if 还不如在下面加一行
    # for i in range(1,len(hypermeter['dZ'])-1):
    # 因为是 BP 所以从大到小
    for i in range(0,len(hypermeter['dZ'])-1):
        bpNumb = len(hypermeter['dZ'])-1 -i;
        # print(bpNumb)
        # hypermeter['dW'][bpNumb] = hypermeter['dZ'][bpNumb]*hypermeter['A'][bpNumb-1].T/m
        hypermeter['dW'][bpNumb] = np.dot(hypermeter['dZ'][bpNumb],hypermeter['a'][bpNumb-1].T)/m + hypermeter['w'][bpNumb]*hypermeter['lambda'][0]/m
        hypermeter['dB'][bpNumb] = np.sum(hypermeter['dZ'][bpNumb],axis = 1)/m
        g = 1.0/(1.0 + np.exp(-hypermeter['z'][bpNumb-1]));
        g = np.multiply(g,(1-g));
        hypermeter['dZ'][bpNumb-1]=np.multiply(np.dot(hypermeter['w'][bpNumb].T,hypermeter['dZ'][bpNumb]),g)
        # print(g)

    hypermeter['dW'][0] = np.dot(hypermeter['dZ'][0],X)/m
    hypermeter['dB'][0] = np.sum(hypermeter['dZ'][0],axis = 1)/m
    hypermeter['w'] = hypermeter['w'] - hypermeter['alpha'][0]*hypermeter['dW']
    hypermeter['b'] = hypermeter['b'] - hypermeter['alpha'][0]*hypermeter['dB']

    print('System ::: BP calculate finished')
    print('================')
# ================== 4.check cv part green ==================
# ================== 5.check train part green ==================
# ================== 6.check test part green ==================
# ================== 7.save mode result and hypermeter green ==================
def saveHypermeters(hypermeter):
    obj = []
    obj.append('hidelayer')
    obj.append('w')
    obj.append('b')
    obj.append('iteration')
    obj.append('alpha')
    obj.append('lambda')
    obj.append('output')

    outputHypermeter = dict()

    for name in obj:
        # print(name,np.array(hypermeter[name]))
        outputHypermeter[name] = hypermeter[name]

    sio.savemat('data/myPicHypermeter.mat',outputHypermeter)




    # \\\\\\\\\\\\ h5py 不能用啊
    # I may be repeating @jpp's answer, but I need work write out this detail to understand what's going on.
    # If I read the problem correctly, Data_set is a list of pairs (lists), each consisting of a 3d array and a single character string.
    # This loop splits it into 2 lists:



    # data = []
    # label = []
    # hf = h5py.File('data/myPicHypermeter.hdf5', 'w')
    # # for i in range(len(Data_set)):
    # for name in obj:
    #     print(name,hypermeter[name])
    #     data.append(hypermeter[name])
    #     label.append(name)
    #     hf.create_dataset(name, data=np.array([hypermeter[name]]), compression='lzf')

    # label = [int(i) for i in label]#convert label to int
        # hf.create_dataset(name, data=np.array(hypermeter[name]), compression='lzf')
    # data = np.array(data)
    # alternatively it could be written as

    # data = [a[0] for a in Data_set]
    # label = [a[1] for a in Data_set]
    # or even

    # data, label = list(zip(*Data_set))
    # When you save data:
    # print(data)
    # with h5py.File('data/myPicHypermeter.h5', 'w') as hf:
    #     hf.create_dataset('data', data=data, compression='lzf')
        # hf.create_dataset('label', data=label, compression='lzf')
    # h5py converts it to an array (it can only save np.array sources).
    # Look at np.array(data).shape. It will be 4d. That looks like a logical data structure for a collection of 3d arrays (identical sized).
    # That could be turned back into a list of 3d arrays, e.g. list(dt).
    # You could do data = np.concatenate(data, axis=0) before the same. That would produce a 3d array, but then you loose all boundaries between the original 3d arrays.

    return outputHypermeter

#=================== tools ===========================
#显示所有 J 的曲线，来展示，我们现在的算法是否已经到达了极限
def showJHistoryMap(J_history):
    J_history = np.array(J_history) #防止 J_history 是一个list
    m= J_history.shape[0]
    print(J_history.shape,"J_history.shape")
    iteration = np.linspace(0,m,m)
    plt.plot(iteration,J_history,'-',linewidth=1)
    plt.show()

def softmaxNumber(activation):
    return np.argmax(np.exp(activation)/np.sum(activation,0),0)






#============= main =============
if __name__ == '__main__':
    # ================== 1.hypermeter and basedata initialize ==================
    dataMat = sio.loadmat('data/ex4data1.mat')

    # 注意这里的假设是你运用的数据图片，已经被调整成为 m = 6000，里面每一个数据源 n = 400 的格式
    # 400 = 20pix * 20pix * 1channel (20*20 的灰度图，通道内的数据是一个灰度)
    X = np.mat(dataMat['X'])
    y = np.mat(dataMat['y'])

    # hypermeter = h5py.File('data/myPicHypermeter.h5', "r")

    #个性化数据
    #[*******************  这里是模型手动输入的地方 *********************************]
    # 这里的y 可以认为是一个10output 啊，所以这里必须是一个
    classssIndex = np.where(y==10)[0] # 找到所有的 10 把他们全部变成0
    y[classssIndex,:] = 0

    # 将 y 根据 type 进行分类,output 将根据这个值进行变化
    ym = y.shape[0]
    output= 10
    yc = np.zeros((output,ym))
    # \\\\\\\\\\\\\\ 这里有木有将算法优化的空间
    for i in range(0,ym):
        yc[int(y[i]),i] = 1
    y = yc.T
    m,n = X.shape
    # 初始化 hyper meter，在这个方法里面也包含了部分全局参数的声明。
    if os.path.exists('data/myPicHypermeter.mat'):
        hypermeter = sio.loadmat('data/myPicHypermeter.mat')
    else:
        hypermeter = None
    # print(hypermeter['iteration'][0])
    # print(hypermeter)
    hypermeter = initializeHypermeters(output,n,hypermeter)
    Xtrain,ytrain,XCV,yCV,Xtest,ytest = initializeData(X,y)
    # ================== 3.check bp function green ==================

    # ================== 5.check train part green ==================
    historyJ = []
    # historyCV = []
    for i in range(0,hypermeter['iteration'][0]):
    # for i in range(0,10):
        FowradPropagationFC(hypermeter,Xtrain,ytrain)
        regularization = np.sum(np.sum(np.array(hypermeter['w'][len(hypermeter['dZ'])-1])**2))*hypermeter['lambda'][0]/2/m
        historyJ.append(np.average(np.abs(hypermeter['a'][len(hypermeter['a'])-1] - ytrain))+regularization)
        # historyCV.append(np.average(np.abs(FowradPropagationFC(hypermeter,XCV,yCV) - yCV))+regularization)
        backPropagationFC(hypermeter,Xtrain,ytrain)


    print('System ::: CV calculate finished')
    print('================')
    # print(np.array(historyJ).shape)
    showJHistoryMap(historyJ)
    # showJHistoryMap(historyCV)

    # ================== 4.check cv part green ==================
    #prodiction
    prodictionReuslt = softmaxNumber(FowradPropagationFC(hypermeter,XCV,yCV))
    yss = np.argmax(yCV,0) - prodictionReuslt
    ym,yn = yss[np.where(yss==0)[0],0].shape
    yCVn,yCVm = yCV.shape
    # print(np.argmax(yCV,0))
    print(ym,yn,yCVm)
    print(ym/yCVm*100,'%')



    # ================== 6.check test part green ==================
    #prodiction
    prodictionReuslt = softmaxNumber(FowradPropagationFC(hypermeter,Xtest,ytest))
    yss = np.argmax(ytest,0) - prodictionReuslt
    ym,yn = yss[np.where(yss==0)[0],0].shape
    ytestn,ytestm = ytest.shape
    # print(np.argmax(ytest,0))
    print(ym/ytestm*100,'%')

    # ================== 7.save mode result and hypermeter green ==================
    saveHypermeters(hypermeter)
    print(hypermeter['w'][0].shape)

    # ================== 6.prodiction picture ==================
    # pictureWidth = 20
    # pictureHight = 20
    # currentPicIndex = 0
    # while True:
    #     orders = input("Type any words: (type quit to quit)")
    #     if(orders=="quit"):
    #         break
    #     Xpicture = Xtest[currentPicIndex,:]
    #
    #     m,n = Xpicture.shape;
    #     prodictionReuslt = softmaxNumber(FowradPropagationFC(hypermeter,Xpicture,ytest))
    #     print("我觉得这个图应该是",prodictionReuslt)
    #
    #     # 取一张照片
    #     gridData0 = Xpicture
    #     # 重新排列成图片的样子
    #     grid0 = gridData0.reshape(pictureWidth,pictureHight).T
    #     fig, ax3 = plt.subplots()
    #     ax3.imshow(grid0,extent = [0,10,0,10],aspect='auto',cmap='gray')
    #     ax3.set_title(' we get a picture')
    #     plt.tight_layout()
    #     plt.show()
    #
    #     currentPicIndex=currentPicIndex+1

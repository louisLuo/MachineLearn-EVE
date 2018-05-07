# 我们对数据的存储使用 pandas
import pandas as pd
 # 数据的分析和使用，我这里使用 numpy
import numpy as np
#我们对数据的展示，使用 pyplot
import matplotlib.pyplot as plt

#引入算法
import linearRegression
import logisticRegression



# ==========================
# 练习例子

# data = pd.read_excel('data.xls', index_col=u'纳税人编号')
# 使用identity 5*5 的矩阵
# print(np.eye(5))
# print(np.zeros((2, 1)) #生成2行1列数据，注意里面要2个括号
# print(a = np.matrix(’1 2; 3 4’)) #这里生成一个矩阵可以使用列用；分开
# a = np.random.randint(0,10,size=(9,9));
# print(a) #一个0-10的随机 9*9 的矩阵
# print(np.power(a,2)) #求矩阵2次幂
# print(a.sum()) #求和
# print(a.sum(axis=0))#0 是纵向求和， 1 是横向求和 结果都会变成横向的一组数组
# print(a.max()) #求最大
# print(a.argmax()) #求最大值的位置
# print(a.mean()) #求平均值
# print(np.row_stack((a,a))) #在列上面加一个矩阵
# print(np.column_stack((a,a))) #在行上面加一个矩阵
# element = a[:,2] #从数据中拿出数据来
# np.transpose(a)#matrix transpose
# print(element)
# a[:,2] = [7 ,8 ,6 ,4 ,9 ,9 ,3 ,3 ,1]#赋值方式
# print(a)

# 获取外部数据
# data = pd.read_csv('ex1data1.CSV')
# print(data.tail)
# data = data.as_matrix()  # 将数据转化为矩阵

# X = data[:,0];
# y = data[:,1];
# X = X[0::,1::]; #对X 进行矩阵截取 操作，这里和 matlab不同，前面一个代表的是行数，逗号后面代表的是列数
# X = np.array(data[:,0]) #创建 np 的一个数组哦
# y = np.array(data[:,1])
# m = len(y)
# print(X*y)# 这种乘法是将对应矩阵位置的数值相乘，而不是矩阵乘法运算。
# print(np.dot(X,y))
# print(a)
# print(a.transpose()) #matrix transport
# print(a**-1)# 逆矩阵
# print(a.reshape(-1,1)) #把横的变成竖的1列矩阵
# print(np.size(a)) #获得数据样本大小
# 绘制图形
# plt.plot(X, y,'go--', linewidth=1, label='ROC of LM')  # 绘制线 这里的go 是目标点绘制成O
# plt.xlabel('False Positive Rate')  # 坐标轴标签
# plt.ylabel('True POstive Rate')
# plt.xlim(0, 25)  # 设定边界范围
# plt.ylim(-5, 25)
# plt.legend(loc=4)  # 设定图例位置
# plt.show()  # 显示绘图结果
#
# plt.plot(X,y,'bo')  # 绘制点
# plt.show()
#
# plt.plot(X,y,'r+')  # 绘制 + 号
# plt.show()

# ==========================
# linear regression

# # 获取外部数据
# data = pd.read_csv('ex1data1.CSV')
# data = data.as_matrix()  # 将数据转化为矩阵
# X = np.array(data[:,0]) #创建 np 的一个数组哦
# y = np.array(data[:,1])
# m = len(y)
#
# # 开始计算cost function
# theta = np.zeros((2, 1))
# theta = np.matrix('-1 ; 2')
#
# # 开始计算批量的 iteration cost function
# iterations = 1500;
# alpha = 0.01;
# # 这个方法中 x 必须在前面全部加1
# Xones = np.column_stack((np.ones((m,1)),X))
# thetaP,Jhistory = linearRegression.linearRegression(Xones,y,theta,alpha,iterations,False)
#
# #测试所有alpha的值
# alpha = np.array([0.3, 0.1, 0.03, 0.01])
# linearRegression.alphaRateTest(Xones,y,theta,alpha,iterations)
#
# # 绘制前我需要将函数的 prodect 结果放进去哦
# plt.plot(X,y,'r+')  # 绘制点
# plt.plot(X,np.dot(Xones,thetaP),'-',LineWidth=1)
# plt.show()


# ==========================
# logistic regression

data = pd.read_csv('ex2data2.CSV') #注意 CSV 的头部部分必须要有，不能开头就是 数字
data = data.as_matrix();
X = np.array(data[:,0:2]);# 到第3列，但是不包括第3列
y = np.array(data[:,2]);
m = len(y);

# ---  制作一个图形现实
positive = np.where(y==1.)  # 获得所有为1 的位置，返回一个矩阵
negative = np.where(y==0)   # 获得所有为0 的位置，返回一个矩阵

# plt.plot(X[positive,0],X[positive,1],'r+') #通过 positive 矩阵查找位置，所以事实上，逗号前后其实就是矩阵么，r+ 表示 red + 号
# plt.plot(X[negative,0],X[negative,1],'bo') #通过 negative 矩阵查找位置，所以事实上，逗号前后其实就是矩阵么，bo 表示 bule o 号
# plt.show()

#按照惯例往 X 里面添加 x0

Xones = np.column_stack((np.ones((m,1)),X))
# print(X)
theta = np.array([1, 1, 1]);
iterations = 1000
lambdas = 10
alpha = 0.05
J, theta = logisticRegression.logisticRegression(Xones,y,theta,alpha,iterations,lambdas,True)
print("the best result ",J)
print(theta)

# # #测试所有alpha的值
# alpha = np.array([0.3, 0.1, 0.03, 0.01])
# logisticRegression.alphaRateTest(Xones,y,theta,alpha,iterations)

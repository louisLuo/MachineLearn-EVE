import numpy as np

# this function is to compute the Cost function return a J
# all data is base on the numpy class ! so this input data should be numpy
def computeCost(X,y,theta):
    #格式化所有数据 y 必须是列的
    y = y.reshape(-1,1);
    m = len(y)
    theta = np.array(theta)

    # first you should all x0 into X !
    X = np.column_stack((np.ones((m,1)),X))
    # thetaT = np.transpose(theta);
    Hx = np.dot(X,theta)
    # print(np.size(Hx))
    # print(np.size(y))
    # print(Hx)
    # print(y)
    J = (np.power((Hx - y),2).sum())/(2*m)
    return J

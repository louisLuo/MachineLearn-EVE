import numpy as np
import h5py
import matplotlib.pyplot as plt
#这里的主要套路是通过 model 来进 iteration

#%matplotlib命令可以将matplotlib的图表直接嵌入到Notebook之中
# %matplotlib inline



    # ================== 1.hypermeter and basedata initialize ==================

    # ================== 2.check fp function green ==================
def conv_forward(A_prev,W,b,hparameters):
    '''
    fucntion description
    implemets the forward propagation for a convolution function

    input:
    A_prev: slice of input data in (f,f,n_C_pre,n_C)
    W: the weight(f,f,n_C_pre,n_C)
    b: the bias(1,1,1,n_C)
    hpermeters: python dictionary contain 'stride' and 'pad'

    return
    Z: a scalar value
    cashe: cashe of values needed for conv backpropagation function
    '''

    # get size that we need like n_H_prev,n_W_prev,n_C
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    #part 1:  calculate the final size
    n_H = int(1+(n_H_prev + 2*pad - f )/stride)
    n_W = int(1+(n_W_prev + 2*pad - f )/stride)

    Z = np.zeros((m,n_H,n_W,n_C))

    #part 2:  calculate the value and insert it
    #prepare
    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for channel in range(n_C):
                    #current postion
                    #we calculate base on the final result matrix size, and get data from original data.
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    # get one cell from source
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    # attetion:: this W and b is filter size !!!!! not of the X or Y
                    Z[i,h,w,channel] = conv_single_step(a_slice_prev,W[:,:,:,channel],b[:,:,:,channel])

    #验证出来的尺寸和预计的尺寸一致
    assert(Z.shape == (m,n_H,n_W,n_C))
    cashe = (A_prev,W,b,hparameters)
    return Z,cashe

def pool_forward(A_prev,hparameters,mode='MAX'):
    '''
    fucntion description
    implemets the forward propagation for a convolution function
    there is no padding here

    input:
    A_prev: slice of input data in (f,f,n_C_pre,n_C)
    hpermeters: python dictionary contain 'stride' and 'pad'
    mode: the mode you create the filter
        MAX: the max result write into matrix
        AVERAGE: the average result write into matrix

    return
    Z: a scalar value
    cashe: cashe of values needed for conv backpropagation function
    '''

    # get size that we need like n_H_prev,n_W_prev,n_C
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    # (f,f,n_C_prev,n_C) = W.shape

    stride = hparameters['stride']
    f = hparameters['f']

    #part 1:  calculate the final size
    n_H = int(1+(n_H_prev  - f )/stride)
    n_W = int(1+(n_W_prev  - f )/stride)
    n_C = n_C_prev

    A = np.zeros((m,n_H,n_W,n_C))

    #part 2:  calculate the value and insert it
    #prepare
    # A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):
        # a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for channel in range(n_C):
                    #current postion
                    #we calculate base on the final result matrix size, and get data from original data.
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    # get one cell from source
                    a_slice_prev = A_prev[vert_start:vert_end,horiz_start:horiz_end,channel]
                    # different mode has different result
                    if mode == 'MAX':
                        A[i,h,w,channel] = np.max(a_slice_prev)
                    elif mode == 'AVERAGE':
                        A[i,h,w,channel] = np.average(a_slice_prev)

    #验证出来的尺寸和预计的尺寸一致
    assert(A.shape == (m,n_H,n_W,n_C))
    cashe = (A_prev,hparameters)
    return A,cashe



    # ================== 3.check bp（model） function green ==================

def conv_backward(dZ,cache):
    '''
    fucntion description
    implemets the backpropagation for a convolution function
    there is no padding here
    公式：
    Z = a*W+b


    dZ = Z
    dA = w*dZ
    dW = a*dZ
    db = dZ

    (a是source data，Z 是result data，w 是 filter)



    input:
    dZ: gradient of the cost with respect to input of the conv layer(A_prev)
    其实就是 conv 求解的 结果Z
    cache:


    return
    dA: gradient of A
    dW: gradient of W
    db: gradient of b
    '''

    (A_prev,W,b,hparameters)= cache

    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    (f,f,n_C_prev,n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m,n_H,n_W,n_C) = dZ.shape

    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))


    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    #current postion
                    #we calculate base on the final result matrix size, and get data from original data.
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    # get one cell from source
                    a_slice = A_prev_pad[i,vert_start:vert_end,horiz_start:horiz_end,:]
                    # attetion:: calculate the driver by cell
                    # Z[i,h,w,channel] = conv_single_step(a_slice_prev,W[:,:,:,channel],b[:,:,:,channel])
                    da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]+=W[:,:,:,c]*dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]

    dA_prev[i,:,:,:] = da_prev_pad[pad:-pad:,pad:-pad,:]
    #验证出来的尺寸和预计的尺寸一致
    assert(dA_prev.shape == (m,n_H_prev,n_W_prev,n_C_prev))

    return dA_prev,dW,db

def create_mask_from_window(x):
    '''
    create a mask from a input matrix x,to identify the max entry of xself.
    本质上 max pool 的filter 就是一个 最大值位置是1 ，其他位置全是0的w啊

    Arguments:
    x -- Array of shape (f,f)

    return:
    mask -- array of the same shape as window, contain a ture at the postion crresponding to the max entry of x.
    '''

    mask = None;
    # print(np.argmax(x))
    mask = (x == np.max(x))
    print('mask',mask)
    return mask;

def distribute_value(dz,shape):
    '''
    distribute the input value in the matrix of demension shape
    本质上 平均值的 filter 2*2 大小的话，就是w 都是 1/4 嘛

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H,n_W) of the output matrix for what we want distribute the value of dz

    return:
    a -- Array of size (n_H,n_W) for which we distributed the value of dz
    '''

    (n_H,n_W) = shape
    average = dz/(n_H*n_W)
    a = np.ones((n_H,n_W))*average
    return a

def pool_backward(dA,cache,mode='MAX'):
    '''
    implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache --  cache output from the FB of the pooling layer
    mode -- the pooling mode you would like to use(MAX,AVERAGE)

    output:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    '''

    (A_prev,hparameters) = cache

    stride = hparameters['stride']
    f = hparameters['f']

    m,n_H_prev,n_W_prev,n_C_prev = A_prev.shape
    m,n_H,n_W,n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f

                    if mode == 'MAX':
                        a_perv_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_perv_slice)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]+=mask*dA[i,h,w,c]

                    elif mode == 'AVERAGE':
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]+=distribute_value(da,shape)

    assert(dA_prev.shape == A_prev.shape)
    return dA_prev

    # ================== 5.check train part green ==================

    # ================== 4.check cv part green ==================

    # ================== 6.check test part green ==================

    # ================== 7.save mode result and hypermeter green ==================

    # ================== 8.prodiction picture ==================

    # ================== 9.tools ==================
def zero_pad(X,pad):
    '''
    fucntion description
    pad a lot zero around the X, to make max_pool has a 'SAME' size like original image;

    input:
    X: original Image
    pad: an image have 4 side, pad one size to add 'pad size of zero' into image

    return
    X_pad:
    '''

    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
    return X_pad


def conv_single_step(a_slice_prev,W,b):
    '''
    fucntion description
    conv2d function to make convolution calculate(single cell)
    we calculate one single slice with filter(W), and back with b,this is a single result
    of one time of formulate.

    the size of filer is  (f,f,channel)

    input:
    a_slice_prev: slice of input data in (f,f,n_C_pre)
    W: the weight(f,f,n_C_pre)
    b: the bias(1,1,1)

    return
    Z: a scalar value this is a sum,not max,not average
    '''

    s = a_slice_prev*W
    Z = np.sum(s)+ np.sum(b)
    # Z = Z
    return Z



#============= main =============
if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (5.0,4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(1)

    x = np.random.randn(4,3,3,2)
    x_pad = zero_pad(x,2)
    print(x_pad)

    #show 2 picture in 1 plt to know how difference bettwen
    fig,axarr = plt.subplots(1,2)
    axarr[0].set_title('X')
    axarr[0].imshow(x[0,:,:,0])
    axarr[1].set_title('X_pad')
    axarr[1].imshow(x_pad[0,:,:,0])
    plt.show()

    #calculate one cell of filter
    np.random.seed(1)
    a_slice_prev = np.random.randn(4,4,3)
    W = np.random.randn(4,4,3)
    b = np.random.randn(1,1,1)

    Z = conv_single_step(a_slice_prev,W,b)
    print('Z=',Z)

    #conv FP
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hpermeters = {'pad':2,'stride':2}
    Z,cashe_conv = conv_forward(A_prev,W,b,hpermeters)
    print('Z mean is',np.mean(Z))

    #Pool FP
    np.random.seed(1)
    A_prev = np.random.randn(2,4,4,3)
    hparameters = {'stride':2,'f':3}
    A,cashe = pool_forward(A_prev,hparameters,'MAX')
    print('Max pool A=',A)
    A,cashe = pool_forward(A_prev,hparameters,'AVERAGE')
    print('Average pool A=',A)

    #conv bp
    np.random.seed(1)
    dA,dW,db = conv_backward(Z,cashe_conv)
    print('dW mean',np.mean(dW))

    #max pool bp
    np.random.seed(1)
    x = np.random.randn(2,3)
    mask = create_mask_from_window(x)
    print(x)

    #average pool bp
    a = distribute_value(2,(2,2))
    print('distribute value =',a)

    # max pool bp and average pool bp
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride" : 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])
    print()
    dA_prev = pool_backward(dA, cache, mode = "average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
# from PIL import Image #好像只能在 python2 上面执行
from scipy import ndimage
from tensorflow.python.framework import ops
from cnn_utils import*
#这里的主要套路是通过 model 来进 iteration




    # ================== 1.hypermeter and basedata initialize ==================
def create_placeholder(n_H0,n_W0,n_C0,n_y):
    '''
    create placeholder for tensorflow

    X 的shape 是  m,h,w,c 就是将c存进了最基本的单位中。m,h,w 都存了尺寸。
    这里的 none部分就是m图的数量，我们唯一不知道的是 X 里面，图的数量到底有多少张，不固定

    Y 的shape 是 m,n_y 这里的 n_y 是我们固定的 Y 的type 种类，也就是output 的部分，我们设定为6

    所有的设定，都是为了创建最后的步骤 model，那么我们在完成model的创建之后，只需要存储这个model就可以
    你如果想使用，只需要放入一些 X或者y 的值就可以
    '''

    X = tf.placeholder(tf.float32,shape=(None,n_H0,n_W0,n_C0))
    Y = tf.placeholder(tf.float32,shape=(None,n_y))

    return X,Y

def initialize_parameters():
    '''
    初始化W 啊 这种东西
    '''

    tf.set_random_seed(1)

    W1 = tf.get_variable('W1',[4,4,3,8],initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2',[2,2,8,16],initializer = tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {'W1':W1,'W2':W2}
    return parameters
    # ================== 2.check fp function green ==================
def forward_propagation(X,parameters):
    '''
    function description:
    CONV2D > RELU > MAXPOOL |> CONV2D > RELU > MAXPOOL |> FLATTEN > FULLYCONNECTED

    input:
    X -- Data_set
    parameters -- W1,W2,W3 sth. like it.

    return:
    the output of the last LINER unit
    '''

    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')

    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

    #这个动作是将多维度的数据转变排列成更少甚至1维度的数列，是full connected的前置动作,比如这里 (?,2,2,16)将被序列化成为(?,64)
    # print(P2.shape)
    P2 = tf.contrib.layers.flatten(P2)
    # print(P2.shape)
    #全面连接嘛，就等于 input ，没有hiderlayer，直接output6这样子的神经网络input = (?,64),output=(?,6)
    Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn=None)
    # print(Z3.shape)
    return Z3

def compute_cost(Z3,Y):
    '''
    function description:
    compute the cost function value

    input:
    X -- Data_set
    parameters -- W1,W2,W3 sth. like it.

    return:
    the output of the last LINER unit
    '''

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels = Y))

    return cost
    # ================== 3.check bp（model） function green ==================
def model(X_train,Y_train,X_test,Y_test,learning_rate=0.009,
    num_epochs=10,minibatch_size=4,print_cost=True):

    '''
    function description:
    CONV2D > RELU > MAXPOOL |> CONV2D > RELU > MAXPOOL |> FLATTEN > FULLYCONNECTED
    to calculate the final wight for

    input:
    X -- Data_set
    parameters -- W1,W2,W3 sth. like it.

    return:
    the output of the last LINER unit
    '''

    #initialize data
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    _,n_y = Y_train.shape
    costs = []

    #do
    # initialize data 创造数据空间
    X,Y = create_placeholder(n_H0,n_W0,n_C0,n_y)
    # initialize hypermeter
    parameters = initialize_parameters()
    # FP
    Z3 = forward_propagation(X,parameters)
    # cost
    cost = compute_cost(Z3,Y)
    # bp
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #ready to loop
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        #batch 技术，通过cpu的多核和多电脑协同进行计算
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatchs = int(m/minibatch_size)
            seed=seed+1
            minibatchs = random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatchs:
                (minibatch_X,minibatch_Y) = minibatch
                _,temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                minibatch_cost += temp_cost/num_minibatchs

            if print_cost == True and epoch%5 == 0:
                print('cost after epoch %i:%f'%(epoch,minibatch_cost))
            if print_cost == True and epoch%1 == 0:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteraion(per 10)')
        plt.title('learning_rate='+str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3,1)
        corrent_prediction = tf.equal(predict_op,tf.argmax(Y,1))

        accuracy = tf.reduce_mean(tf.cast(corrent_prediction,'float'))
        print(accuracy)
        # 这里有时候会报错，比如 不能使用 tensorflow 转换 eval 什么的，因为这个代码必须放在with session 里面。
        train_accuracy = accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy = accuracy.eval({X:X_test,Y:Y_test})
        print(train_accuracy,test_accuracy)

        return train_accuracy,test_accuracy,parameters


    # ================== 5.check train part green ==================

    # ================== 4.check cv part green ==================

    # ================== 6.check test part green ==================

    # ================== 7.save mode result and hypermeter green ==================

    # ================== 6.prodiction picture ==================



#============= main =============
if __name__ == '__main__':
    # %matplotlib inline #这里的% 表示什么？ \\\\\\\\
    # ================== 0.insert baseData and check ==================
    np.random.seed(1)
    X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

    index = 6
    # plt.imshow(X_train_orig[index])
    # plt.show()
    # plt.show()
    print('y=',str(np.squeeze(Y_train_orig[:,index])))


    X_train = X_train_orig/255 #里面是0-255的数字，除以255，让他的值变成0-1之间算事 标准化过程
    X_test = X_test_orig/225
    # print(np.eye(6)[Y_train_orig.reshape(-1)].T) 从np.eye 中找到数列代替Y里面的数字。
    Y_train = convert_to_one_hot(Y_train_orig,6).T
    Y_test = convert_to_one_hot(Y_test_orig,6).T
    conv_layers = {}

    # 创建 tensorflow 的 X，Y 元数据用 placeholder
    X,Y = create_placeholder(64,64,3,6)
    print(str(X),str(Y))

    # 创建 tensorflow 用的，W参数
    tf.reset_default_graph()
    with tf.Session() as asess_test:
        parameters = initialize_parameters()
        init = tf.global_variables_initializer()
        asess_test.run(init)
        print('W1=',str(parameters['W1'].eval()[1,1,1]))
        print('W2=',str(parameters['W2'].eval()[1,1,1]))


    # FP
    tf.reset_default_graph()
    with tf.Session() as sess:
        np.random.seed(1)
        X,Y = create_placeholder(64,64,3,6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X,parameters)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(Z3,{X:np.random.randn(2,64,64,3),Y:np.random.randn(2,6)})
        print('Z3=',str(a))

    # compute the cost function
    tf.reset_default_graph()
    with tf.Session() as sess:
        np.random.seed(1)
        X,Y = create_placeholder(64,64,3,6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X,parameters)
        cost = compute_cost(Z3,Y)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = sess.run(cost,{X:np.random.randn(4,64,64,3),Y:np.random.randn(4,6)})
        print('cost = ',str(a))

    #bp
    _,_,parameters = model(X_train,Y_train,X_test,Y_test)

    #prediction pic
    fname = 'datasets/thumbs_up.jpg'
    image = np.array(ndimage.imread(fname,flatten=False))
    my_image = scipy.misc.imresize(image,size=(64,64))
    plt.imshow(my_image)
    plt.show()

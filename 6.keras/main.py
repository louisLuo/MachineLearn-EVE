'''
keras 是基于 theano 的一个深度学习的框架，它的设计参考了torch，用python编写，是一个高度模块化的神经网络库，支持GPU和CPU


本模型是用来分析 来到我家的人是不是迎着笑脸，如果是，就开门，如果不是，就不开门这样。所以叫 happy house～～～
'''
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# ================== 1.hypermeter and basedata initialize ==================




# ================== 2.check fp function green ==================
# ================== 3.check bp function green or model function  green==================
def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    only the happy face man can be inside

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

     ###Input ###
    # Input层参数解释：
    #  （1）‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量
    #  （2） ‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量
    X_input = Input(shape=input_shape)

    ###  Zero-Padding: pads the border of X_input with zeroes ###
    #  ZeroPadding2D 层参数解释：
    # padding：整数tuple，表示在要填充的轴的起始和结束处填充0的数目
    X = ZeroPadding2D(padding=(3, 3))(X_input)

    ###CONV -> BN -> RELU Block applied to X ###
    #Conv2D参数解释：
    #  （1)filters：卷积核的数目（即输出的维度）
    # （2）kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
    # （3） strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。
    X = Conv2D(filters=32,kernel_size=(3, 3), strides = (1, 1), name = 'conv0')(X)

    ###（批）规范化BatchNormalization ：该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1###
    # BatchNormalization层参数解释：
    # axis: 整数，指定要规范化的轴，通常为特征轴（此处我理解为channels对应的轴）。
    # 例如在进行data_format="channels_first"的2D卷积后，一般会设axis=1；例如在进行data_format="channels_last"的2D卷积后，一般会设axis=3
    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    ###Activation层###
    # Activation层 参数解释：
    # activation：将要使用的激活函数，为预定义激活函数名或一个Tensorflow/Theano的函数
    X = Activation('relu')(X)

    ###MAXPOOL层 ###
    # MAXPOOL层参数解释：
    # pool_size：整数或长为2的整数tuple，代表在两个方向（(vertical, horizontal)）上缩小其维度，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为两个维度值相同且为该数字。
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool')(X)  # pool_size=(2, 2) 表示将使图片在rows，cols两个维度上均变为原长的一半

    ###Flatten层###
    # Flatten层参数解释：
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)  #Flatten层用来将输入“压平”，即把多维的输入一维化

    ### Dense层 ###
    # Dense层参数解释：
    # Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
    #（1） units：大于0的整数，代表该层的输出维度
    # （2）activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
    X = Dense(units=1, activation='sigmoid', name='fc')(X)   # 1 指代表该层的输出维度

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    ### END CODE HERE ###

    return model

# ================== 5.check train part green ==================

# ================== 4.check cv part green ==================

# ================== 6.check test part green ==================

# ================== 7.save mode result and hypermeter green ==================

# ================== 8.prodiction picture ==================

# ================== 9.tools ==================


if __name__ == '__main__':
    # initialize data
    X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

    X_train = X_train_orig/255
    X_test = X_test_orig/255

    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print('number of train examples',X_train.shape[0])
    print('number of test examples',X_test.shape[0])

    #bp with Model
    happyModel = HappyModel(X_train.shape[1:])
    happyModel.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    happyModel.fit(x=X_train,y=Y_train,epochs=10,batch_size=32)

    preds = happyModel.evaluate(x=X_test,y=Y_test)
    print('loss',str(preds[0]))
    print('accuracy',str(preds[1]))

    #predic
    ### START CODE HERE ###
    img_path = 'images/my_image.jpg'
    ### END CODE HERE ###
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(happyModel.predict(x))

    #tools
    happyModel.summary()

    # down program have to install grapphviz and pydot
    # plot_model(happyModel, to_file='HappyModel.png')
    # SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))

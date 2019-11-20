
from keras.layers import Conv2D,MaxPooling2D,Dropout,UpSampling2D,Input,concatenate,Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import BatchNormalization,Deconv3D,Conv3D,ReLU,MaxPooling3D
from sklearn.decomposition import PCA
from keras import regularizers
import numpy as np

#data_name= "PaviaU"
#data_name = "Indian_pines"                            
#data_name = "Salinas_valley"
data_name= "Huston_dataset"
if data_name == "PaviaU": 
    size0=(1,1,6)
    stride0=(1,1,3)
    filter1_4=256
    filter3_4=512
    filter3_5=256 
    input_size=(10,10,103,1)
    classnum=10
if data_name == "Indian_pines": 
    size0=(1,1,10)
    stride0=(1,1,5)
    filter1_4=128
    filter3_4=25
    filter3_5=256 
    input_size=(3,3,220,1)
    classnum=17
if data_name == "Salinas_valley": 
    size0=(1,1,10)
    stride0=(1,1,5)
    filter1_4=256
    filter3_4=512
    filter3_5=256 
    input_size=(6,6,204,1)
    classnum=17
if data_name == "Huston_dataset": 
    size0=(1,1,6)
    stride0=(1,1,3)
    filter1_4=256
    filter3_4=512
    filter3_5=512
    input_size=(6,6,144,1)
    classnum=16
def unet(pretrained_weights = None,input_size = input_size):
    inputs = Input(input_size)
    strides1 = (1, 1, 1)
    strides2 = (1, 1, 2)
    conv00 = Conv3D(64, size0,strides=stride0,dim_ordering='tf', activation=None,padding='valid', kernel_initializer = 'he_normal')(inputs)
    conv00 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv00)
    conv00 = ReLU()(conv00)    
    conv1 = Conv3D(64, (3,3,3),strides=(1,1,2),dim_ordering='tf', activation=None,padding='same', kernel_initializer = 'he_normal')(conv00)
    conv1 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv1)
    conv1 = ReLU()(conv1)   
    conv2 = Conv3D(128, (3,3,3),strides=strides2,dim_ordering='tf', activation=None,padding='same', kernel_initializer = 'he_normal')(conv1)    
    conv2= BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv2)
    conv2= ReLU()(conv2)
    conv3 = Conv3D(256, (3,3,3), strides=strides2,dim_ordering='tf', activation=None,padding='same', kernel_initializer = 'he_normal')(conv2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv3)
    conv3 = ReLU()(conv3)
    conv4 = Conv3D(filter1_4, (3,3,3), strides=strides2,dim_ordering='tf' ,activation=None,padding='same', kernel_initializer = 'he_normal')(conv3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv4)
    conv4 = ReLU()(conv4)
    conv5 = Conv3D(64, (1,1,3), strides=strides2,dim_ordering='tf', activation=None,padding='same', kernel_initializer = 'he_normal')(conv00) 
    conv5 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv5)
    conv5 = ReLU()(conv5)    
    conv6 = Conv3D(128, (1,1,3), strides=strides2,dim_ordering='tf', activation=None,padding='same', kernel_initializer = 'he_normal')(conv5) 
    conv6 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv6)
    conv6 = ReLU()(conv6)  
    conv7 = Conv3D(256, (1,1,3), strides=strides2,dim_ordering='tf', activation=None,padding='same', kernel_initializer = 'he_normal')(conv6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv7)
    conv7 = ReLU()(conv7) 
    conv8 = Conv3D(filter3_4, (1,1,3), strides=strides2,dim_ordering='tf' ,activation=None,padding='same', kernel_initializer = 'he_normal')(conv7)
    conv8 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv8)
    conv8 = ReLU()(conv8)    
    conv9 = Conv3D(filter3_5, (1,1,3), strides=strides1,dim_ordering='tf' ,activation=None,padding='same', kernel_initializer = 'he_normal')(conv8)
    conv9 = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06,center=True, scale=True)(conv9)
    conv9 = ReLU()(conv9)    
    conv9 = concatenate([conv4,conv9],axis=4)
    conv10= Conv3D(classnum, (1,1,1),strides =strides1, activation = 'softmax'
                   , padding = 'same', kernel_initializer = 'he_normal',name='conv_')(conv9)
    model = Model(input=inputs, output=conv10)
    model.summary()
    if(pretrained_weights):
        	model.load_weights(pretrained_weights)
    return model

















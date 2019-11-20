

import os
import numpy as np
from model3d import unet
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report
import pandas as pd

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
chose_learning = 'train'
model_num = "3D"
nb_epoch =150
#data_name= "PaviaU"
#data_name = "Indian_pines"                            
#data_name = "Salinas_valley"
data_name= "Huston_dataset"
if data_name == "PaviaU": 
    block_size = 11
    patch_size = 10
    classnum = 10
    target_names = ['c1', 'c2', 'c3', 'c4','c5', 'c6', 'c7', 'c8', 'c9','c10']
if data_name == "Indian_pines": 
    block_size = 4
    patch_size = 3
    classnum = 17
    target_names = ['c1', 'c2', 'c3', 'c4','c5', 'c6', 'c7', 'c8', 'c9','c10', 'c11', 'c12', 'c13','c14', 'c15',"c16"]
if data_name == "Salinas_valley": 
    block_size = 7
    patch_size = 6
    classnum = 17
    target_names = ['c1', 'c2', 'c3', 'c4','c5', 'c6', 'c7', 'c8', 'c9','c10', 'c11', 'c12', 'c13','c14', 'c15',"c16"]
if data_name == "Huston_dataset": 
    block_size = 7
    patch_size = 6
    classnum = 16
    target_names = ['c1', 'c2', 'c3', 'c4','c5', 'c6', 'c7', 'c8', 'c9','c10', 'c11', 'c12', 'c13','c14', 'c15']     
def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
def focal_loss_fixed(y_true, y_pred):
    gamma = 2.0
    alpha = 1.0
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)
     
def scheduler(epoch):
    if epoch % 35== 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
         
if chose_learning=='train':
    X_train = np.load("./predata/x0trainwindowsize" + str(patch_size) + str(data_name) + ".npy")
    train_label = np.load("./predata/y0trainwindowsize" + str(patch_size) + str(data_name) + ".npy")
    X_test = np.load("./predata/x0testwindowsize" + str(patch_size) + str(data_name)+  ".npy")
    test_label = np.load("./predata/y0testwindowsize" + str(patch_size) + str(data_name) + ".npy")
    X_val = np.load("./predata/x0valwindowsize" + str(patch_size) + str(data_name)+  ".npy")
    val_label = np.load("./predata/y0valwindowsize" + str(patch_size) + str(data_name) + ".npy")    
    X_train = X_train[ :,:,:,:,np.newaxis]
    X_val = X_val[ :,:,:,:,np.newaxis]
    X_test = X_test[ :,:,:,:,np.newaxis]

    model = unet()
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                  loss=focal_loss(alpha=1), metrics=['categorical_accuracy'])   
    model_checkpoint = ModelCheckpoint('./models/'+str(model_num)+str(data_name)+str(block_size)+'-'+str(patch_size)+'.hdf5',
                                       monitor='val_categorical_accuracy',verbose=1,save_best_only= True)
    csvlogger=CSVLogger('./record/log_'+str(model_num)+str(data_name)+str(block_size)+'-'+str(patch_size)+'.csv', separator=',', append=True)
    earlystopping=EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    reduce_lr = LearningRateScheduler(scheduler)
    callback_lists = [model_checkpoint,csvlogger,earlystopping,reduce_lr]
    history = model.fit(X_train, train_label, batch_size=64, nb_epoch=nb_epoch,
             verbose=1, shuffle=True, validation_data=(X_val, val_label),callbacks=callback_lists)

    
    def reports(X_test,y_test):
        Y_pred = model.predict(X_test)
        print(Y_pred.shape)
        arg = []
        test= []
        max_pred = []
        max_test = []
        correctnum = 0
        nozero = 0
        for batch0 in range(Y_pred.shape[0]):
            for h0 in range(Y_pred.shape[1]):
                for w0 in range(Y_pred.shape[2]):
                    max_pred = []
                    max_test = []
                    for n in range(0,classnum):
                        max_pred.append(0)
                        max_test.append(0)
                    for c0 in range(Y_pred.shape[3]):
                        Y_pred[batch0,h0,w0,c0,0] =0                    
                        max_pred = np.sum([Y_pred[batch0,h0,w0,c0,:],max_pred],axis =0)                       
                        max_test = np.sum([y_test[batch0,h0,w0,c0,:],max_test],axis =0)
                    i = np.argmax(max_pred)
                    j = np.argmax(max_test)                
                    if j!=0:
                        nozero = nozero + 1
                    if j!=0 and i == j :
                        correctnum = correctnum+1
                    arg.append(i)
                    test.append(j)
        if data_name == "PaviaU": 
            classification = classification_report(arg,test,target_names = target_names,digits=4,labels=[1,2,3,4,5,6,7,8,9])
        if data_name == "Huston_dataset": 
            classification = classification_report(arg,test,target_names = target_names,digits=4,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        if data_name == "Indian_pines" or "Salinas_valley": 
            classification = classification_report(arg,test,target_names = target_names,digits=4,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])                
        totalavg = correctnum / nozero
        print('correct_num : ',correctnum)
        print("pixel_num :",nozero)
        return classification,totalavg
        
    # show current path
    PATH = os.getcwd()
    print (PATH)
    model = load_model('./models/'+str(model_num)+str(data_name)+str(block_size)+'-'+str(patch_size)+'.hdf5',
                       custom_objects={'focal_loss_fixed':focal_loss_fixed})
    print('.......waiting.........')   
    classification , totalavg= reports(X_test,test_label)
    
    print(str(data_name)+"__classification result: ")
    print('{}'.format(classification))
    print(str(data_name)+"__classification OA: ")
    print('{}'.format(totalavg))




#loss accuracy image
    import matplotlib.pyplot as plt 
    fig = plt.figure()
    plt.plot(history.history['categorical_accuracy'],label='training acc') 
    plt.plot(history.history['val_categorical_accuracy'],label='val acc') 
    plt.title('Accuracy')
    plt.ylabel('Accuracy(%)') 
    plt.xlabel('epoch')
    plt.legend(loc='lower right') 
    fig.savefig('./record/'+str(model_num)+str(data_name)+str(block_size)+'-'+str(patch_size)+'acc.png') 
    fig = plt.figure() 
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss') 
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('./record/'+str(model_num)+str(data_name)+str(block_size)+'-'+str(patch_size)+'loss.png')
    fig = plt.figure() 





















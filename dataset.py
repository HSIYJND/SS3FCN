# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:54:55 2019

@author: admin
"""

import os
from typing import Tuple
import numpy as np
from random import shuffle, randint
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import scipy.io as sio
from keras.utils import to_categorical
import random
import scipy
from keras.layers import concatenate
from sklearn.decomposition import PCA,KernelPCA
from spectral import *

A = 3 #agument time
data_name = "Huston_dataset"   #"PaviaU"# "Indian_pines"#"Salinas_valley"#"Huston_dataset"
fold_num_train = 0   #num of folds choose for train and Val
#paviaU
if data_name == "PaviaU":
    data_name = "PaviaU"                             
    datapath ="./data/PaviaU.mat"
    ground_truth_path = "./data/PaviaU_gt.mat"
    window_size = 11
    batch_size =10
    allclass_num = 10 #9+1class_num = 9
    channels = 103
    numlabel = 3
    width = 340
    height = 610
    class_num = 9
    num_fold = 10
    fold_num_val=int(fold_num_train+num_fold/2)
#indianpines
if data_name == "Indian_pines":     
    data_name = "Indian_pines"                             
    datapath ="./data/Indian_pines.mat"
    ground_truth_path = "./data/Indian_pines_gt.mat"
    window_size = 4
    batch_size =3
    allclass_num = 17 #16+1
    channels = 220
    numlabel =3
    width = 145
    height = 145
    class_num = 16
    num_fold = 4
    fold_num_val=int(fold_num_train+num_fold/2)
if data_name == "Salinas_valley": 
    data_name = "Salinas_valley"                             
    datapath ="./data/Salinas_valley.mat"
    ground_truth_path = "./data/Salinas_valley_gt.mat"
    window_size = 7
    batch_size = 6
    allclass_num = 17 #16+1
    channels = 204
    numlabel = 3
    width = 217
    height = 512
    class_num =16
    num_fold = 9#9
    fold_num_val=int(fold_num_train+num_fold/2)
if data_name == "Huston_dataset": 
    data_name = "Huston_dataset"                             
    datapath ="./data/huston.mat"
    ground_truth_path = "./data/huston_gt.mat"
    window_size = 7
    batch_size = 6
    allclass_num = 16 #15+1
    channels = 144
    numlabel = 3
    width = 1905
    height = 349
    class_num =15
    num_fold =6
    fold_num_val=int(fold_num_train+num_fold/2)
class WindowSize(object):
    def __init__(self, x: int, y: int):
        if not x > 0 or not y > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x, y))
        elif not isinstance(x, int) or not isinstance(y, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(type(x),
                                                                      type(y)))
        self.x = x
        self.y = y


class Stride(object):
    def __init__(self, x_stride: int, y_stride: int):
        if not x_stride > 0 or not y_stride > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x_stride,
                                                                  y_stride))
        elif not isinstance(x_stride, int) or not isinstance(y_stride, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(
                    type(x_stride),
                    type(y_stride)))
        self.x = x_stride
        self.y = y_stride


class Patch:
    def __init__(self,
                 index: int,
                 left_x: int,
                 right_x: int,
                 upper_y: int,
                 lower_y: int):
        self.index = index
        self.left_x = left_x
        self.right_x = right_x
        self.upper_y = upper_y
        self.lower_y = lower_y

def sliding_window(image: np.ndarray, window_size: WindowSize,
                   stride: Stride = 0):
    number_of_patches_in_x = int(((image.shape[0] - window_size) / stride) + 1)    
    number_of_patches_in_y = int(((image.shape[1] - window_size) / stride) + 1)
    patches = []
    index = 0
    for x_dim_patch_number in range(0, number_of_patches_in_x):
        for y_dim_patch_number in range(0, number_of_patches_in_y):
            left_border_x = int(0 + stride * x_dim_patch_number)
            right_border_x = int(window_size + stride * x_dim_patch_number)
            upper_border_y = int(0 + stride * y_dim_patch_number)
            lower_border_y = int(window_size + stride * y_dim_patch_number)                            
            patch = Patch(index, upper_border_y, lower_border_y,left_border_x, right_border_x)
            patches.append(patch)
            index += 1     
    return patches

def extract_grids(dataset_path: str, ground_truth_path: str, window_size: int, batch_size):    
    data_path = os.path.join(os.getcwd(), 'data')
    if data_name == "PaviaU":
         input_data = sio.loadmat(os.path.join(data_path,'PaviaU.mat'))['paviaU']
         input_data0 = sio.loadmat(os.path.join(data_path,'PaviaU.mat'))['paviaU']
         gt0 = sio.loadmat(os.path.join(data_path,'PaviaU_gt.mat'))['paviaU_gt']
         gt1 = sio.loadmat(os.path.join(data_path,'PaviaU_gt.mat'))['paviaU_gt']
    #indianpines
    if data_name == "Indian_pines":
        input_data = sio.loadmat(os.path.join(data_path,'Indian_pines.mat'))['indian_pines']
        input_data0 = sio.loadmat(os.path.join(data_path,'Indian_pines.mat'))['indian_pines']
        gt0 = sio.loadmat(os.path.join(data_path,'Indian_pines_gt.mat'))['indian_pines_gt']
        gt1 = sio.loadmat(os.path.join(data_path,'Indian_pines_gt.mat'))['indian_pines_gt']    
#        print('inputdata'+str(type(input_data)))
    
    if data_name == "Salinas_valley":
        input_data = sio.loadmat(os.path.join(data_path,'Salinas_valley.mat'))['salinas_corrected']
        input_data0 = sio.loadmat(os.path.join(data_path,'Salinas_valley.mat'))['salinas_corrected']
        gt0 = sio.loadmat(os.path.join(data_path,'Salinas_valley_gt.mat'))['salinas_gt']  
        gt1 = sio.loadmat(os.path.join(data_path,'Salinas_valley_gt.mat'))['salinas_gt']                  
       
    if data_name == "Huston_dataset":
        input_data = sio.loadmat(os.path.join(data_path,'huston.mat'))['huston']
        input_data0 = sio.loadmat(os.path.join(data_path,'huston.mat'))['huston']
        gt0 = sio.loadmat(os.path.join(data_path,'huston_gt.mat'))['huston_gt']  
        gt1 = sio.loadmat(os.path.join(data_path,'huston_gt.mat'))['huston_gt']     
    ground_truth = imshow(data=input_data, figsize=(5, 5))  
    ground_truth2 = imshow(classes=gt0.astype(int), figsize=(5, 5))
  
    gt = [gt0 for K in range(numlabel)]
    gt = np.reshape (gt,(numlabel,height,width))
    gt = np.transpose(gt,(1,2,0))

    patches = sliding_window(input_data, window_size,window_size)
    test_pixels=0
    total = 0
    diff_patch = []
    same_patch = []
    train_patches = []
    train_patches_gt = []
    val_patches = []
    val_patches_gt = []    
    test_patches = []
    test_patches_gt = [] 
    a=[]
    b=[]
    i=0
    for num in range(0,class_num+1):
            a.append(0)
    for num in range(0,class_num+1):
            b.append(0)            
    for patch in patches:
        nonzero = np.count_nonzero(gt0[patch.upper_y:patch.lower_y,
                                      patch.left_x:patch.right_x])        
        if nonzero == 0:
            continue  
        total += nonzero                             
        ex0 = gt0[patch.upper_y:patch.lower_y,patch.left_x:patch.right_x].copy()
        
        flatten_ = ex0.flatten()
        length_ = list(set(flatten_))
        if len(length_) > 1:
            diff_patch.append(patch)       
        if len(length_) == 1:            
            same_patch.append(patch)   
    for newpatch in diff_patch:
        if i% num_fold ==fold_num_train:
            i = i+1
            train_patch = input_data[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x, :].copy()                        
            train_patch_gt = gt[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x,:].copy()
            gt[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x,:]=0
            
            ex1 = gt0[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x].copy()
            for x in range(0,ex1.shape[0]):
                for y in range(0,ex1.shape[1]):
                    for z in range(1,class_num+1):
                        if ex1[x,y] == z :                                
                            a[z] = a[z] +1
            newpatches1 = sliding_window(train_patch,batch_size,1)            
            for patch1 in newpatches1:
                train = train_patch[patch1.upper_y:patch1.lower_y,patch1.left_x:patch1.right_x, :].copy()                
                train_gt = train_patch_gt[patch1.upper_y:patch1.lower_y,patch1.left_x:patch1.right_x,:].copy()                                    
                train_patches.append(train)
                train_patches_gt.append(train_gt)
                train_patches,train_patches_gt = AugmentData(train,train_gt,train_patches,train_patches_gt)
            gt1[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x] = 26
            input_data0[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x,:] = 26                                                 
        elif i%num_fold==fold_num_val:
            i = i+1
            val_patch = input_data[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x, :].copy()            
            val_patch_gt = gt[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x,:].copy()
            gt[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x,:]=0
            ex2 = gt0[newpatch.upper_y:newpatch.lower_y,newpatch.left_x:newpatch.right_x].copy()
            for x in range(0,ex2.shape[0]):
                for y in range(0,ex2.shape[1]):
                    for z in range(1,class_num+1):
                        if ex2[x,y] == z :                                
                            b[z] = b[z] +1            
            newpatches3 = sliding_window(val_patch,batch_size,1)            
            for patch3 in newpatches3:
                val = val_patch[patch3.upper_y:patch3.lower_y,patch3.left_x:patch3.right_x, :].copy()                
                val_gt = val_patch_gt[patch3.upper_y:patch3.lower_y,patch3.left_x:patch3.right_x,:].copy()                    
                val_patches.append(val)
                val_patches_gt.append(val_gt)
        else:
            i=i+1
    all_test_patches = sliding_window(input_data,batch_size,batch_size)
    for patch in all_test_patches:
        nonzero = np.count_nonzero(gt[patch.upper_y:patch.lower_y,
                              patch.left_x:patch.right_x])        
        if nonzero == 0:
            continue 
        test_patch = input_data[patch.upper_y:patch.lower_y,patch.left_x:patch.right_x, :].copy()            
        test_patch_gt = gt[patch.upper_y:patch.lower_y,patch.left_x:patch.right_x,:].copy()
        for i in range(test_patch_gt.shape[0]):
            for j in range(test_patch_gt.shape[1]):
                if test_patch_gt[i,j,0]!=0:
                    test_pixels=test_pixels+1                    
        test_patches.append(test_patch)
        test_patches_gt.append(test_patch_gt) 
    print("test_pixels:", test_pixels)                               
    return train_patches, train_patches_gt, val_patches,val_patches_gt,test_patches, test_patches_gt, gt1,input_data0,a,b
       


def savePreprocessedData(path, X_train, X_val,X_test,y_train,y_val, y_test, windowSize):      
    data_path = os.path.join(os.getcwd(), path)
    
    with open(os.path.join(data_path, "x0trainwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
            np.save(outfile, X_train)   
    with open(os.path.join(data_path, "x0valwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
            np.save(outfile, X_val)
    with open(os.path.join(data_path, "x0testwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
            np.save(outfile, X_test)
   

    with open(os.path.join(data_path, "y0trainwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
            np.save(outfile, y_train) 
    with open(os.path.join(data_path, "y0valwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
        np.save(outfile, y_val)    
    with open(os.path.join(data_path, "y0testwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
            np.save(outfile, y_test) 
#  

            
def AugmentData(X_train,y_train,extracted_patches,extracted_patches_gt):
    for k in range(0,A):
        num = random.randint(0,2)
        if (num == 0):            
            flipped_patch = np.flipud(X_train)
            flipped_patch_gt = np.flipud(y_train)
        if (num == 1):             
            flipped_patch = np.fliplr(X_train)
            flipped_patch_gt = np.fliplr(y_train)
        if (num == 2):            
            no = random.randrange(-180,180,30)
            flipped_patch = scipy.ndimage.interpolation.rotate(X_train, 
                            no,axes=(1, 0), reshape=False, output=None, 
                            order=3, mode='constant', cval=0.0, prefilter=False)
            flipped_patch_gt = scipy.ndimage.interpolation.rotate(y_train, 
                            no,axes=(1, 0), reshape=False, output=None, 
                            order=3, mode='constant', cval=0.0, prefilter=False)
        extracted_patch = flipped_patch
        extracted_patch_gt = flipped_patch_gt
        extracted_patches.append(extracted_patch)
        extracted_patches_gt.append(extracted_patch_gt)
    return extracted_patches, extracted_patches_gt


          

train_patches, train_patches_gt, val_patches,val_patches_gt ,test_patches, test_patches_gt,datashow,datashowor,a ,b= extract_grids(datapath, ground_truth_path, window_size,batch_size)
ground_truth = imshow(classes=datashow.astype(int), figsize=(5,5))
print("train-----"+str(np.array(train_patches).shape))
print("traingt-----"+str(np.array(train_patches_gt).shape))
print("val-----"+str(np.array(val_patches).shape))
print("valgt-----"+str(np.array(val_patches_gt).shape))
print("test-----"+str(np.array(test_patches).shape))
print("testgt-----"+str(np.array(test_patches_gt).shape))


train_patches_gt = to_categorical(train_patches_gt,allclass_num)
val_patches_gt = to_categorical(val_patches_gt,allclass_num)
test_patches_gt = to_categorical(test_patches_gt,allclass_num)
print("testgt-----"+str(np.array(test_patches_gt).shape))
savePreprocessedData("predata",train_patches,val_patches,test_patches,train_patches_gt,val_patches_gt,test_patches_gt,batch_size)





print("train pixels each class num:",a)
print("train pixels num:",np.sum(a))
print("Val pixels each class num:",b)



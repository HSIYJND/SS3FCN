![image](https://github.com/leonzx7/SS3FCN/blob/master/img-storage/figoverlap.png)
Training-test information leakage problem:
	As shown in Figure 1(b), both patches represent ‘3’. If one of them is selected as training data and the other one is selected as the test data, the evaluation results cannot demonstrate the real discrimination ability of the obtained model. Although the spatial features is vital for HSI analysis, the previous usage of spatial information in patch-wise classification might be inappropriate.

![image](https://github.com/leonzx7/SS3FCN/blob/master/img-storage/fig2.png)
Data partitioning method：
	we split the multi-class blocks in Salinas Valley into 9 folds and the order of samples in the k-th fold is [k, 9+k, . . . , 9N+k] where N represents the number of samples in this fold. We select a single fold as the training set, the other one as the validation set, and the remaining 7 folds as the test. See article for details

![image](https://github.com/leonzx7/SS3FCN/blob/master/img-storage/fig5.png)
	The SS3FCN architecture used for HSI semantic segmentation. (a) the basic unit of SS3FCN; (b) and (c) the structure
of the proposed SS3FCN. Prior to the fusion of two branches, we employ a convolution layer with large stride (1*1*10 for
Salinas Valley and Indian Pines, 1*1*6 for Pavia University and Houston University). The features from 3D branch and 1D
branch are concatenated into 512 feature maps, and forwarded to the last convolution layer.
	


Methods of training and testing SS3FCN:

	1. Modify data_name,fold_num_train in dataset. Py
	
	2. Change data_name in model3d.py
	
	3. Change data_name in main.py

Important training Settings:

	1.python 3.6
	
	2. Tensorflow - gpu 1.9.0
	
	3. Keras 2.2.4
	
	4.spectral 0.19
	
	5. Other basic python package
	
Note: due to the limitation of upload file size, you need to put the dataset into the "data" folder	
	

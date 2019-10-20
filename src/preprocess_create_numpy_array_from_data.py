import numpy as np 
import keras
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
from medpy import metric
import numpy as np
import os
from keras import losses
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np 
import nibabel as nib
import cv2
global_counter=0
import matplotlib.pyplot as plt 
from skimage import data, exposure
# update the path to load data set all the files into one folder
path_for_data="/home/raun/liver_stuff/TrainingBatch2/LITS/TrainingBatch2/"
#update the path to store the images as png file to check the preprocessing and fix if not satisfactory
path_to_visualize_the_preprocessed_files="/home/raun/liver_stuff/X_test_whole_check/"
path_to_save_data="/home/raun/liver_stuff/" 

def find_min_and_max(img):
	hist = cv2.calcHist([img.flatten()],[0],None,[256],[1,256])
	max_to_compare=np.amax(hist)
	#print(hist)
	#print(max_to_compare)
	index_list=[]
	value_list=[]
	for i in range(0,256):
		index_list.append(i)
		value_list.append((float)(hist[i]))

	fig,ax=plt.subplots()
	ax.plot(index_list,value_list)
	ax.set(title='get_range',xlabel='X',ylabel='Y')
	point1,point2=fig.ginput(2)
	print(point1,point2)
	point1,point2=(int)(point1[0]),(int)(point2[0])
	ax.axvspan(point1,point2,color='red',alpha=0.5)
	fig.canvas.draw()
	plt.show()

	img[img<point1]=point1
	img[img>point2]=point2
	#img= exposure.equalize_hist(img)


	return img


to_write_counter=0
counter=0
a=[]
b=[]

for i in range(0,131):
	a=[]
	b=[]
	print i
	counter+=1
	X_train=nib.load(path_for_data+"volume-"+str(i)+".nii").get_data()
	#X_train[X_train<-200]=0
	#X_train[X_train>256]=1
	X_train=((X_train-np.amin(X_train))*1.0)/(np.amax(X_train)-np.amin(X_train))
	X_train=(np.swapaxes(X_train,0,-1))

	
	X_train=X_train*255
	X_train=X_train.astype(np.uint8)
	#cv2.imwrite("test.png",X_train[40])
	X_train=(((find_min_and_max(X_train))))
	

	print(np.amax(X_train),np.amin(X_train))

	
	if(i>52 and i<69):
		for xxii in range(0,len(X_train)):
			X_train[xxii]=np.flipud(X_train[xxii])
			
	elif(i>82 and i<131):

		for xxii in range(0,len(X_train)):
			X_train[xxii]=np.flipud(X_train[xxii])
			
	elif(i>67 and i<83):
		for xxii in range(0,len(X_train)):
			X_train[xxii]=np.fliplr(X_train[xxii])
	for j in range(0,len(X_train)):

		a.append(cv2.equalizeHist(X_train[j,:,:]))

	for j in range(0,len(a)):
		cv2.imwrite(path_to_visualize_the_preprocessed_files+str(i)+"_"+str(j)+".png",a[j])


	


	np.save(path_to_save_data+"X_train_"+str(i)+".npy",a)
	print(np.array(a).shape)




	mask=nib.load(path_for_data+"segmentation-"+str(i)+".nii").get_data()

	mask=(np.swapaxes(mask,0,-1))

	y_train_liver=np.zeros((mask.shape),dtype=np.float32)
	

	y_train_liver[mask>0]=1
	

	X_liver=a*y_train_liver
	print(np.amax(X_liver),np.amin(X_liver))


	#print(np.array(X_train).shape)
	#y_liver
	
	if(i>52 and i<69):
		for xxii in range(0,len(X_liver)):
			
			X_liver[xxii]=np.flipud(X_liver[xxii])
	elif(i>82 and i<131):

		for xxii in range(0,len(X_liver)):
			
			X_liver[xxii]=np.flipud(X_liver[xxii])
	elif(i>67 and i<83):
		for xxii in range(0,len(X_liver)):
			
			X_liver[xxii]=np.fliplr(X_liver[xxii])
	for j in range(0,len(X_liver)):
		b.append((X_liver[j,:,:]))


	np.save(path_to_save_data+"X_train_liver"+str(i)+".npy",b)


	b=[]


	
	
	
	mask=nib.load(path_for_data+"segmentation-"+str(i)+".nii").get_data()

	mask=(np.swapaxes(mask,0,-1))

	y_train_liver=np.zeros((mask.shape),dtype=np.float32)
	
	y_train_liver[mask>0]=1
	print("liver_input",np.amax(y_train_liver),np.amin(y_train_liver))


	
	if(i>52 and i<69):
		for xxii in range(0,len(y_train_liver)):
			
			y_train_liver[xxii]=np.flipud(y_train_liver[xxii])
		
	elif(i>82 and i<131):

		for xxii in range(0,len(y_train_liver)):
		
			y_train_liver[xxii]=np.flipud(y_train_liver[xxii])
		
	elif(i>67 and i<83):
		for xxii in range(0,len(y_train_liver)):
			
			y_train_liver[xxii]=np.fliplr(y_train_liver[xxii])
	for j in range(0,len(y_train_liver)):

		b.append(y_train_liver[j,:,:])
	


	np.save(path_to_save_data+"y_train_liver"+str(i)+".npy",b)

	b=[]


	

	mask=nib.load(path_for_data+"segmentation-"+str(i)+".nii").get_data()

	mask=(np.swapaxes(mask,0,-1))

	
	y_train_lesion=np.zeros((mask.shape),dtype=np.float32)
	
	y_train_lesion[mask>1]=1
	print(np.amax(y_train_lesion),np.amin(y_train_lesion))
	#y_train_lesion[mask<=1]=0


	


	#print(np.array(X_train).shape)
	#y_liver
	
	if(i>52 and i<69):
		for xxii in range(0,len(y_train_lesion)):
			
			y_train_lesion[xxii]=np.flipud(y_train_lesion[xxii])
		
	elif(i>82 and i<131):

		for xxii in range(0,len(y_train_lesion)):
			
			y_train_lesion[xxii]=np.flipud(y_train_lesion[xxii])
		
	elif(i>67 and i<83):
		for xxii in range(0,len(y_train_lesion)):
			
			y_train_lesion[xxii]=np.fliplr(y_train_lesion[xxii])
	for j in range(0,len(y_train_lesion)):
		b.append(y_train_lesion[j,:,:])



	np.save(path_to_save_data+"y_train_tumor"+str(i)+".npy",b)

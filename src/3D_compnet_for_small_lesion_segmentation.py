import keras
from keras import optimizers
#from keras.utils import multi_gpu_model
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
from medpy import metric
import numpy as np
import os
from keras import losses
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv3D, MaxPooling3D, Activation, UpSampling3D,Dropout,Conv3DTranspose,add,multiply
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
#from keras.applications import Xception
from keras.utils import multi_gpu_model
import random
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import nibabel as nib
CUDA_VISIBLE_DEVICES = [1,2,3,4]
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])

import numpy as np
import cv2

smooth = 1.
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_test(y_true, y_pred):

    y_true_f = np.array(y_true).flatten()
    y_pred_f =np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def neg_dice_coef_loss(y_true, y_pred):
    return dice_coef(y_true, y_pred)
def Comp_U_net(input_shape,learn_rate=1e-3):

    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape,name='ip0')



    

    conv0a = Conv3D( 64, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    
    
    conv0a = bn()(conv0a)
    
    conv0b = Conv3D(64, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv0a)

    conv0b = bn()(conv0b)

    

    
    pool0 = MaxPooling3D(pool_size=(2, 2, 2))(conv0b)

    pool0 = Dropout(DropP)(pool0)




    conv1a = Conv3D( 128, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool0)
    
    
    conv1a = bn()(conv1a)
    
    conv1b = Conv3D(128, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv1a)

    conv1b = bn()(conv1b)


    
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1b)

    pool1 = Dropout(DropP)(pool1)



    

    conv2a = Conv3D(256, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
    
    conv2a = bn()(conv2a)

    conv2b = Conv3D(256, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv2a)

    conv2b = bn()(conv2b)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2b)

    pool2 = Dropout(DropP)(pool2)






    conv5b = Conv3D(512, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)

    conv5b = bn()(conv5b)

    



    up6 = concatenate([Conv3DTranspose(256,(2, 2,2), strides=(2, 2,2), padding='same')(conv5b), (conv2b)],name='up6', axis=3)


    up6 = Dropout(DropP)(up6)

    conv6a = Conv3D(256, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
    
    conv6a = bn()(conv6a)

    





    up7 = concatenate([Conv3DTranspose(128,(2, 2,2), strides=(2, 2,2), padding='same')(conv6a),(conv1b)],name='up7', axis=3)

    up7 = Dropout(DropP)(up7)
    #add second output here

    conv7a = Conv3D(128, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up7)
    
    conv7a = bn()(conv7a)

 

   




    up8 = concatenate([Conv3DTranspose(64,(2, 2,2), strides=(2, 2,2), padding='same')(conv7a), (conv0b)],name='up8', axis=3)

    up8 = Dropout(DropP)(up8)
 
    conv8a = Conv3D(64, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up8)
    
    conv8a = bn()(conv8a)

    
    
    final_op=Conv3D(1, (1, 1,1), activation='sigmoid',name='final_op')(conv8a)
    


    #----------------------------------------------------------------------------------------------------------------------------------

    #second branch - brain
    xup6 = concatenate([Conv3DTranspose(256,(2, 2,2), strides=(2, 2,2), padding='same')(conv5b), (conv2b)],name='xup6', axis=3)

    

    xup6 = Dropout(DropP)(xup6)

    xconv6a = Conv3D(256, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup6)
    
    xconv6a = bn()(xconv6a)

    




    xup7 = concatenate([Conv3DTranspose(128,(2, 2,2), strides=(2, 2,2), padding='same')(xconv6a),(conv1b)],name='xup7', axis=3)

    xup7 = Dropout(DropP)(xup7)
    
    xconv7a = Conv3D(128, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup7)
    
    xconv7a = bn()(xconv7a)




    xup8 = concatenate([Conv3DTranspose(64,(2, 2,2), strides=(2, 2,2), padding='same')(xconv7a),(conv0b)],name='xup8', axis=3)

    xup8 = Dropout(DropP)(xup8)
    #add third xoutxout here
    
    xconv8a = Conv3D(64, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup8)
    
    xconv8a = bn()(xconv8a)



   
    
    xfinal_op=Conv3D(1, (1, 1,1), activation='sigmoid',name='xfinal_op')(xconv8a)


    #-----------------------------third branch



    #Concatenation fed to the reconstruction layer of all 3
   
    x_u_net_op0=keras.layers.concatenate([final_op,xfinal_op,keras.layers.add([final_op,xfinal_op])],name='res_a')
    

    
    #multiply with input






    res_1_conv0a = Conv3D( 64, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(x_u_net_op0)
    
    
    res_1_conv0a = bn()(res_1_conv0a)
    
    res_1_conv0b = Conv3D(64, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv0a)

    res_1_conv0b = bn()(res_1_conv0b)

    res_1_pool0 = MaxPooling3D(pool_size=(2, 2, 2))(res_1_conv0b)

    res_1_pool0 = Dropout(DropP)(res_1_pool0)




    res_1_conv1a = Conv3D( 128, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool0)
    
    
    res_1_conv1a = bn()(res_1_conv1a)
    
    res_1_conv1b = Conv3D(128, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv1a)

    res_1_conv1b = bn()(res_1_conv1b)

    res_1_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(res_1_conv1b)

    res_1_pool1 = Dropout(DropP)(res_1_pool1)



    

    res_1_conv2a = Conv3D(256, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool1)
    
    res_1_conv2a = bn()(res_1_conv2a)

    res_1_conv2b = Conv3D(256, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv2a)

    res_1_conv2b = bn()(res_1_conv2b)

    
    res_1_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(res_1_conv2b)

    res_1_pool2 = Dropout(DropP)(res_1_pool2)




    res_1_conv5b = Conv3D(512, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool2)

    res_1_conv5b = bn()(res_1_conv5b)




    res_1_up6 = concatenate([Conv3DTranspose(256,(2, 2,2), strides=(2, 2,2), padding='same')(res_1_conv5b), (res_1_conv2b)],name='res_1_up6', axis=3)


    res_1_up6 = Dropout(DropP)(res_1_up6)

    res_1_conv6a = Conv3D(256, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up6)
    
    res_1_conv6a = bn()(res_1_conv6a)



    res_1_up7 = concatenate([Conv3DTranspose(128,(2, 2,2), strides=(2, 2,2), padding='same')(res_1_conv6a),(res_1_conv1b)],name='res_1_up7', axis=3)

    res_1_up7 = Dropout(DropP)(res_1_up7)
    #add second res_1_output here
    res_1_conv7a = Conv3D(128, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up7)
    
    res_1_conv7a = bn()(res_1_conv7a)



    res_1_up8 = concatenate([Conv3DTranspose(64,(2, 2,2), strides=(2, 2,2), padding='same')(res_1_conv7a),(res_1_conv0b)],name='res_1_up8', axis=3)

    res_1_up8 = Dropout(DropP)(res_1_up8)
    #add third outout here
    res_1_conv8a = Conv3D(64, (kernel_size, kernel_size,kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up8)
    
    res_1_conv8a = bn()(res_1_conv8a)



    res_1_final_op=Conv3D(1, (1, 1,1), activation='sigmoid',name='res_1_final_op')(res_1_conv8a)






    model=Model(inputs=[inputs],outputs=[final_op,
                                    
                                      xfinal_op,
                                 
                                      res_1_final_op,
                                     
                                    ])
                                      #res_2_final_op,
                                      #res_2_xfinal_op,
                                      #res_3_final_op,])
    #sgd = optimizers.SGD(lr=0.01, decay=1e-8, momentum=0.8, nesterov=True)
    model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),loss={'final_op':neg_dice_coef_loss,
                                                'xfinal_op':dice_coef_loss,
                                                'res_1_final_op':'mse'})
                                                #'res_2_final_op':neg_dice_coef_loss,
                                                #'res_2_xfinal_op':dice_coef_loss,
                                                #'res_3_final_op':'mse'})
    print(model.summary())
    return model
#model=UNet(input_shape=(384,384,1))
model=Comp_U_net(input_shape=(32,32,32,1))

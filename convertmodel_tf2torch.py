import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Activation, Flatten
from tensorflow.keras.layers import Dropout, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam,RMSprop
import time       

# Step 1: Load TensorFlow/Keras model
path = '/home/wennan/Workspace/DeepPETmodified/' 
tf_model = tf.keras.models.load_model(path+'DeepPETtrained.h5')
tf_model.summary()

# Step 2: Define PyTorch model structure based on the TensorFlow model architecture
# Emm.. Not a straightforward task to convert a sequential model to pytorch using ONNX (tf2onnx and onnx2pytorch)
''' 
def deepPETmodel2(learning_rate):
    
    start_time = time.time()
    print ('Constructing Model ... ')
        
    no_angles = 128
    img_size  = 128
    model = Sequential()

    
    model.add(Conv2D(input_shape=(no_angles,img_size,1), filters=32, kernel_size=(7,7), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(7,7), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
                
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
   
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    #model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')) ###
    #model.add(BatchNormalization())
    model.add(Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #model.add(Reshsape((img_size, img_size)))
    
    print ('Model constructed in {0} seconds'.format(time.time() - start_time))
    start_time = time.time()      
        
    #adam = Adam(learning_rate=learning_rate,epsilon=None, decay=0.00001)
    #model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    
    rms = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rms, metrics=['mean_squared_error'])
    
    print ('Model compiled in {0} seconds'.format(time.time() - start_time))
    
    #filepath="weights.best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    return model
        
tf_model = deepPETmodel2(0.0001) 
'''     
class PyTorchDeepPETModel(nn.Module):
    def __init__(self):
        super(PyTorchDeepPETModel, self).__init__()

        # Define layers analogous to the TensorFlow model
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv17 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(256)

        self.conv18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn18 = nn.BatchNorm2d(256)

        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn19 = nn.BatchNorm2d(256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv20 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn20 = nn.BatchNorm2d(128)

        self.conv21 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(128)

        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm2d(128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv23 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn23 = nn.BatchNorm2d(64)

        self.conv24 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn24 = nn.BatchNorm2d(64)

        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))

        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))

        x = self.up1(x)
        x = F.relu(self.bn17(self.conv17(x)))
        x = F.relu(self.bn18(self.conv18(x)))
        x = F.relu(self.bn19(self.conv19(x)))

        x = self.up2(x)
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))

        x = self.up3(x)
        x = F.relu(self.bn23(self.conv23(x)))
        x = F.relu(self.bn24(self.conv24(x)))

        x = self.conv_out(x)
        return x

# Step 3: Initialize PyTorch model
pytorch_model = PyTorchDeepPETModel()

# Step 4: Extract weights from TensorFlow model and set them in PyTorch
def set_weights(pytorch_layer, keras_weights):
    # Convert the Keras weights to PyTorch format
    pytorch_layer.weight.data = torch.from_numpy(np.transpose(keras_weights[0], (3, 2, 0, 1))).float()
    pytorch_layer.bias.data = torch.from_numpy(keras_weights[1]).float()

# Map TensorFlow weights to PyTorch layers
set_weights(pytorch_model.conv1, tf_model.layers[0].get_weights())
set_weights(pytorch_model.conv2, tf_model.layers[3].get_weights())
set_weights(pytorch_model.conv3, tf_model.layers[6].get_weights())
set_weights(pytorch_model.conv4, tf_model.layers[9].get_weights())
set_weights(pytorch_model.conv5, tf_model.layers[12].get_weights())
set_weights(pytorch_model.conv6, tf_model.layers[15].get_weights())
set_weights(pytorch_model.conv7, tf_model.layers[18].get_weights())
set_weights(pytorch_model.conv8, tf_model.layers[21].get_weights())
set_weights(pytorch_model.conv9, tf_model.layers[24].get_weights())
set_weights(pytorch_model.conv10, tf_model.layers[27].get_weights())
set_weights(pytorch_model.conv11, tf_model.layers[30].get_weights())
set_weights(pytorch_model.conv12, tf_model.layers[33].get_weights())
set_weights(pytorch_model.conv13, tf_model.layers[36].get_weights())
set_weights(pytorch_model.conv17, tf_model.layers[40].get_weights())
set_weights(pytorch_model.conv18, tf_model.layers[43].get_weights())
set_weights(pytorch_model.conv19, tf_model.layers[46].get_weights())
set_weights(pytorch_model.conv20, tf_model.layers[50].get_weights())
set_weights(pytorch_model.conv21, tf_model.layers[53].get_weights())
set_weights(pytorch_model.conv22, tf_model.layers[56].get_weights())
set_weights(pytorch_model.conv23, tf_model.layers[60].get_weights())
set_weights(pytorch_model.conv24, tf_model.layers[63].get_weights())
set_weights(pytorch_model.conv_out, tf_model.layers[66].get_weights())

print(pytorch_model)

# https://pytorch.org/tutorials/advanced/cpp_export.html
# Create an example input tensor with the same size as the input to the model
example_input = torch.rand(1, 1, 256, 256)  # Batch size = 1, Channels = 1, Height = 256, Width = 256

# Convert the PyTorch model to TorchScript using tracing
traced_model = torch.jit.trace(pytorch_model, example_input)

# Save the TorchScript model to a file
traced_model.save(path+'deep_pet_traced_model_256.pt')

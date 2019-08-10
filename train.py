#!/usr/bin/env python3  
from keras.datasets import mnist  
from keras.utils import np_utils  
import numpy as np  
import os
np.random.seed(10)  


#######################################################
# Constant
#######################################################
DATA_SOURCES = 'datas\mnist'
SERIALIZED_MODEL_NAME = 'keras_mnist_cnn.h5'
USE_SM_IF_EXIST = True


  
###################################
# Reading Data
###################################
# Read MNIST data  
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()
    
# Translation of data  
X_Train40 = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')  
X_Test40 = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')

# Standardize feature data  
X_Train40_norm = X_Train40 / 255  
X_Test40_norm = X_Test40 /255  
  
# Label Onehot-encoding  
y_TrainOneHot = np_utils.to_categorical(y_Train)  
y_TestOneHot = np_utils.to_categorical(y_Test)

###################################
# Define Model
###################################
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import load_model
  
model = Sequential()  

# Create CN layer 1  
model.add(Conv2D(filters=16,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu'))  

# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2,2)))  
    
# Create CN layer 2  
model.add(Conv2D(filters=36,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu'))  
      
# Create Max-Pool 2  
model.add(MaxPooling2D(pool_size=(2,2)))  
        
# Add Dropout layer  
model.add(Dropout(0.25))  

# Flatten
model.add(Flatten())

# Hidden layer
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.5))  

# Output layer
model.add(Dense(10, activation='softmax'))

# Show Model summary
model.summary()  
print("")  

###################################
# Start Training
###################################
# Define training meta data
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
  
# Start training
if os.path.isfile(SERIALIZED_MODEL_NAME) and USE_SM_IF_EXIST:
    print('Loading model from serialized file={}...'.format(SERIALIZED_MODEL_NAME))
    model = load_model(SERIALIZED_MODEL_NAME)
else:
    train_history = model.fit(x=X_Train40_norm,  
                              y=y_TrainOneHot, validation_split=0.2,  
                              epochs=10, batch_size=300, verbose=2)
    
    
###################################
# Evaluation
###################################
scores = model.evaluate(X_Test40_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

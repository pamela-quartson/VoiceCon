from keras.models import Sequential
from keras.layers.core import Dense
import os
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Trainer:
    def __init__(self,train_data,encodedLabels):
        print('Adding GPUs...')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        self.train_data = train_data
        self.encodedLabels = encodedLabels

        NET = Sequential()
        NET.add(Dense(units = self.train_data.shape[1],input_dim = self.train_data.shape[1],activation = 'relu'))
        NET.add(Dense(units = 512,activation = 'relu'))
        NET.add(Dense(units = 1024,activation = 'relu'))
        NET.add(Dense(units = 512,activation = 'relu'))
        NET.add(Dense(units = 256,activation = 'relu'))
        NET.add(Dense(units = 64,activation = 'relu'))
        NET.add(Dense(units = self.encodedLabels.shape[1],activation = 'softmax'))
                
        NET.summary()
        NET.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
        NET.fit(x = train_data,y = encodedLabels,validation_split= 0.1,epochs = 100,steps_per_epoch=10,validation_steps=1)
        print('Saving Trained Model')
        NET.save('voiceCon_NET.hdf5')
   
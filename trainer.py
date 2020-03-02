from keras.models import Sequential
from keras.layers.core import Dense
import os
import sys


class Trainer:
    def __init__(self,train_data,encodedLabels,see_history = False):

        print('Adding GPUs...')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        self.train_data = train_data
        self.encodedLabels = encodedLabels
        self.see_history = see_history
        self.history = None

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
        self.history = NET.fit(x = train_data,y = encodedLabels,validation_split= 0.1,epochs = 20,steps_per_epoch=10,validation_steps=1)
        print('Saving Trained Model')
        NET.save('voiceCon_NET.hdf5')

        if self.see_history:
            self.plot_history(self.history)

    def plot_history(self,history):
        import matplotlib.pyplot as plt
        print(history)
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


   
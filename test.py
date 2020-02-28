import sister
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pickle
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#================= Get data ============================
door_nv = pd.read_csv('door_nv.txt')
door_pv = pd.read_csv('door_pv.txt')
garage_nv = pd.read_csv('garage_nv.txt')
garage_pv = pd.read_csv('garage_pv.txt')
lights_nv = pd.read_csv('lights_nv.txt')
lights_pv = pd.read_csv('lights_pv.txt')
water_nv = pd.read_csv('water_nv.txt')
water_pv = pd.read_csv('water_pv.txt')

data_ = [door_nv,door_pv,garage_nv,garage_pv,lights_nv,lights_pv,water_nv,water_pv]
#================= PREPROCESS===================================
def make_labels(data):
    print('Making labels ...')
    label_list = []
    label = data.columns.values[0]
    for i in range(len(data)):
        label_list.append(label)
    return pd.DataFrame(label_list)

def preprocess_1(data_to_process):
    print('\n Preprocessing data')
    column_label = data_to_process.columns.values[0]
    df = pd.DataFrame(map(embedder,data_to_process[column_label]))
    return df

def merger(data):
    print('\n Merging data')
    df = preprocess_1(data)
    label = make_labels(data)
    final_df = pd.concat([df,label],axis = 1)
    return final_df

def INIT():
    all_data = pd.DataFrame({}) #init dataframe
    for i in data_:
        d = merger(i)
        all_data = pd.concat([all_data,d])
    all_data = shuffle(all_data)

    with open('data_embedded.txt','w+') as D:
        D.write(str(all_data))
        D.close()

    with open('voiceConData.pickle','wb') as D:
        print('\nWriting Data to directory')
        pickle.dump(all_data,D)
    
#=================================================================================================
def get_data():
     with open('voiceConData.pickle','rb') as D:
         print('\nGetting data from directory')
         data  = pickle.load(D)
         return data

def encode_labels(labels):
    #encode labels
    print('Encoding training labels')
    #for i in labels:print(i)
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    labels = encoder.fit_transform(train_labels)
    LABELS = np_utils.to_categorical(labels)
    #for i in LABELS:print(i)
    return LABELS #encoded labels


      


if __name__ == '__main__':

    embedder = None
    if not os.path.exists('voiceConData.pickle'):
        #get data, preprocess and train
        print('Training data not found. Initializing embedder')
        embedder = sister.MeanEmbedding(lang = 'en')
        INIT()

    sample = get_data()
    #print(sample,sample.shape)
    #reset columns and indices
    sample.columns = list(range(0,sample.shape[1]))
    sample.index = list(range(0,sample.shape[0]))

    train_labels = sample[300]
    sc = MinMaxScaler()
    
    #===========Normalize features=======================
    train_samples = sample[list(range(0,sample.shape[1]-1))]
    train_data = sc.fit_transform(train_samples)
    encodedLabels = encode_labels(train_labels)
    for (a,b) in zip(sample[300],encodedLabels):print(a,b)
    predictor = dict(zip(sample[300],encodedLabels))
    #print(predictor)
    #print(encodedLabels,encodedLabels.shape)
    if not os.path.exists('voiceCon_NET.hdf5'):#if there is no trained model in dir 
        print('NO trained model found for this data, Initializing training on data: VoiceCon')
        NET = Sequential()
        NET.add(Dense(units = train_data.shape[1],input_dim = train_data.shape[1],activation = 'relu'))
        NET.add(Dense(units = 512,activation = 'relu'))
        NET.add(Dense(units = 1024,activation = 'relu'))
        NET.add(Dense(units = 512,activation = 'relu'))
        NET.add(Dense(units = 256,activation = 'relu'))
        NET.add(Dense(units = 64,activation = 'relu'))
        NET.add(Dense(units = encodedLabels.shape[1],activation = 'softmax'))
                
        NET.summary()
        NET.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
        NET.fit(x = train_data,y = encodedLabels,validation_split= 0.1,epochs = 1000,steps_per_epoch=10,validation_steps=1)
        NET.save('voiceCon_NET.hdf5')
   
    #=========== load model and predict ============================
    from keras.models import load_model
    trained_model = load_model('voiceCon_NET.hdf5')
    print('Initializing embedder') 
    embedder = sister.MeanEmbedding(lang = 'en')
    ar = embedder('open the door')
    ar = ar.reshape(1,-1) #reshaping for a single sample
    ar = sc.fit_transform(ar)  
    prediction = trained_model.predict_classes(ar)
    for predicted_label,its_array in predictor.items():
        if its_array[prediction] == 1:
            print('This is it:::::::::::',predicted_label)
            break
   
    
    





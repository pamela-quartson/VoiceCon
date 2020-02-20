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

embedder = None
if not os.path.exists('voiceConData.pickle'):
    print('Training data not found. Initializing embedder')
    embedder = sister.MeanEmbedding(lang = 'en')


'''label = ''
train_labels = []
path = os.getcwd()
for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith('.txt'):
            label = file.rstrip('.txt')
            
            train_labels.append(label)'''

door_nv = pd.read_csv('door_nv.txt')
door_pv = pd.read_csv('door_pv.txt')
garage_nv = pd.read_csv('garage_nv.txt')
garage_pv = pd.read_csv('garage_pv.txt')
lights_nv = pd.read_csv('lights_nv.txt')
lights_pv = pd.read_csv('lights_pv.txt')
water_nv = pd.read_csv('water_nv.txt')
water_pv = pd.read_csv('water_pv.txt')

data_ = [door_nv,door_pv,garage_nv,garage_pv,lights_nv,lights_pv,water_nv,water_pv]

def make_labels(data):
    label_list = []
    label = data.columns.values[0]
    for i in range(len(data)):
        label_list.append(label)
    return pd.DataFrame(label_list)

def preprocess_1(data_to_process):
    column_label = data_to_process.columns.values[0]
    df = pd.DataFrame(map(embedder,data_to_process[column_label]))
    return df

def merger(data):
    df = preprocess_1(data)
    label = make_labels(data)
    final_df = pd.concat([df,label],axis = 1)
    return final_df

def save_dat():
    all_data = pd.DataFrame({}) #init dataframe
    for i in data_:
        d = merger(i)
        all_data = pd.concat([all_data,d])
    all_data = shuffle(all_data)

    with open('data_embedded.txt','w+') as D:
        D.write(str(all_data))
        D.close()

    with open('voiceConData.pickle','wb') as D:
        pickle.dump(all_data,D)
    

def get_data():
     with open('voiceConData.pickle','rb') as D:
         data  = pickle.load(D)
         return data

def encode_labels(labels):
     #encode labels
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    labels = encoder.fit_transform(train_labels)
    LABELS = np_utils.to_categorical(labels)
    return LABELS



if __name__ == '__main__':
    sample = get_data()
    #reset columns and indices
    sample.columns = list(range(0,301))
    sample.index = list(range(0,sample.shape[0]))

    train_labels = sample[300]
    sc = MinMaxScaler()
    

    train_samples = sample[[0,299]]
    train_data = sc.fit_transform(train_samples)
    encodedLabels = encode_labels(train_labels)
    #print(encodedLabels,encodedLabels.shape)
    
    NET = Sequential()
    NET.add(Dense(units = train_data.shape[1],input_dim = train_data.shape[1],activation = 'relu'))
    NET.add(Dense(units = 512,activation = 'relu'))
    NET.add(Dense(units = 1024,activation = 'relu'))
    NET.add(Dense(units = 512,activation = 'relu'))
    NET.add(Dense(units = 256,activation = 'relu'))
    NET.add(Dense(units = 64,activation = 'relu'))
    NET.add(Dense(units = encodedLabels.shape[1],activation = 'relu'))
             
    NET.summary()
    NET.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    NET.fit(x = train_data,y = encodedLabels,validation_split= 0,epochs = 1000)






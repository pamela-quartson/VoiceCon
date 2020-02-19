import sister
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pickle
embedder = sister.MeanEmbedding(lang = 'en')


label = ''
train_labels = []
path = os.getcwd()
for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith('.txt'):
            label = file.rstrip('.txt')
            
            train_labels.append(label)

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
    

print(all_data)

    






#encode labels
encoder = LabelEncoder()
encoder.fit(train_labels)
labels = encoder.fit_transform(train_labels)
LABELS = np_utils.to_categorical(labels)


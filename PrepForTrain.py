import pickle
import os
import pandas as pd
import sister
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
import trainer

class Prep:
    def __init__(self):
        self.CATEGORY_FILE = 'exCats.pickle'
        self.CATEGORIES = None
        self.data_categories = 'category' #categorized data path used for training
        self.prepared_data = 'voiceConData.pickle'
        self.retrain = False
        self.trained_model = 'voiceCon_NET.hdf5'
        
        if not os.path.exists(self.trained_model):
            print('No existing trained model found. Preparing to train new model')
            self.retrain = True
            if not os.path.exists(self.CATEGORY_FILE):
                print('There are no existing categories of commands\n\
                Creating Category file')
                with open(self.CATEGORY_FILE,'wb') as Cfile:
                    pickle.dump([],Cfile) # initialize with empty list
                print('New Categroy file created')
            #after new file is created, check categroy dir and load
            
            print('Checking for new categories...')
            with open(self.CATEGORY_FILE,'rb') as Cfile:
                self.CATEGORIES = pickle.load(Cfile)
                
            
            new_cats = []
            for roots,files,dirs in os.walk(self.data_categories):
                print('Current Categories found: ',dirs)
                for category in dirs:
                    if category not in self.CATEGORIES:
                        new_cats.append(category)
                        print('found new categroy:',category)
                        self.CATEGORIES.append(category)
                        print('Adding new categroy')
            
            if len(new_cats) == 0:print('No new categories found')
            else:
                self.retrain = True
                print('Remebering new Categories:...',new_cats)
        
            with open(self.CATEGORY_FILE,'wb') as Cfile:
                pickle.dump(self.CATEGORIES,Cfile)
            print('Current categories in memory:',self.CATEGORIES)
            
            #NB for now no ability to delete categories
            #i think i will leave it like this for a while

            #now load data and prepcess
            self.data = {}
            for category in self.CATEGORIES:
                #print(category)
                label = category.strip('.txt')
                self.data[label] = pd.read_csv(self.data_categories+'/'+category)
            #print(self.data)
            
            if not os.path.exists(self.prepared_data):
                print('No prepared data for training found')
                print('Initializing Embedder...')
                self.embedder = sister.MeanEmbedding(lang = 'en')
                self.INIT()
            
            if self.retrain:
                print('Since new categories were added or new model required, Retraining model...')
                print('Loading Prepared Data')
                
                output = self.normalize()
                trainingData = output[0]
                labels = output[1]
                trainer.Trainer(trainingData,labels,see_history=True)
        



    def INIT(self):
        all_data = pd.DataFrame({}) #init dataframe
        for i in self.data.values():
            d = self.merger(i)
            all_data = pd.concat([all_data,d])
        all_data = shuffle(all_data)

        with open('data_embedded.txt','w+') as D:
            D.write(str(all_data))
            D.close()

        with open('voiceConData.pickle','wb') as D:
            print('\nWriting Data to directory')
            pickle.dump(all_data,D)
            D.close()
    
    def merger(self,data):
        print('\n Merging data')
        df = self.preprocess(data)
        label = self.make_labels(data)
        final_df = pd.concat([df,label],axis = 1)
        return final_df

    def preprocess(self,data_to_process):
        print('\n Preprocessing data')
        column_label = data_to_process.columns.values[0]
        df = pd.DataFrame(map(self.embedder,data_to_process[column_label]))
        return df
    
    def make_labels(self,data):
        print('Making labels ...')
        label_list = []
        label = data.columns.values[0]
        for i in range(len(data)):
            label_list.append(label)
        return pd.DataFrame(label_list)
    
    def encode_labels(self,labels):
        #encode labels
        print('Encoding training labels')
        #for i in labels:print(i)
        encoder = LabelEncoder()
        encoder.fit(labels)
        labels = encoder.fit_transform(labels)
        LABELS = np_utils.to_categorical(labels)
        #for i in LABELS:print(i)
        return LABELS #encoded labels
    
    def get_data(self):
        with open('voiceConData.pickle','rb') as D:
            print('\nGetting data from directory')
            data  = pickle.load(D)
            return data
                
    def normalize(self):    
        sample = self.get_data()
        #print(sample,sample.shape)
        #reset columns and indices
        sample.columns = list(range(0,sample.shape[1]))
        sample.index = list(range(0,sample.shape[0]))

        train_labels = sample[300]
        sc = MinMaxScaler()
        print('Normalizing Features...')
        #===========Normalize features=======================
        train_samples = sample[list(range(0,sample.shape[1]-1))]
        train_data = sc.fit_transform(train_samples)
        encodedLabels = self.encode_labels(train_labels)
        p = self.make_label_matches(encodedLabels,train_labels)
        print(p)
        
        return train_data,encodedLabels
    
    def make_label_matches(self,encoded,default):
        match = []
        for en in encoded:
            match.append(en.argmax())#append the index of the prediction encoded label
        labels = list(default)
        prediction_matches = dict(zip(match,labels))
        with open('prediction_matches.pickle','wb') as M:
            pickle.dump(prediction_matches,M)

        return prediction_matches


#Prep()


from keras.models import load_model
import PrepForTrain
import os
import sister
from sklearn.preprocessing import MinMaxScaler
import pickle

class Predictor:
    def __init__(self):
        self.prediction_model = 'voiceCon_NET.hdf5'
        self.prediction_matches  = 'prediction_matches.pickle'
        self.matcher = None

        if not os.path.exists(self.prediction_model):
            #if no trained model exists
            print('Trained model not found... Preparing to train new model')
            PrepForTrain.Prep()

        else:print('Trained model found')
        self.embedder = sister.MeanEmbedding(lang = 'en')
        self.NET = load_model(self.prediction_model)

        print('Looking for Prediction Matcher...')
        with open(self.prediction_matches,'rb') as P:
            self.matcher = pickle.load(P)
        print('Prediction Matcher Found!!')
        

    
    def tell(self,to_predict):
        embedded = self.embedder(to_predict)
        embedded = embedded.reshape(-1,1) # reshape for scaling
        sc = MinMaxScaler()
        embedded = sc.fit_transform(embedded)
        embedded = embedded.reshape(1,-1) #reshape for one sample and prediction
        prob,p_class = self.NET.predict(embedded),self.NET.predict_classes(embedded)
        return prob,self.matcher[p_class[0]],self.NET.predict_classes(embedded)



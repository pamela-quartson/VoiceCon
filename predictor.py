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
        embedded = embedded.reshape(1,-1)#reshaping for one sample
        sc = MinMaxScaler()
        embedded_t = sc.fit_transform(embedded)
        print(self.NET.predict(embedded_t))
        prob,p_class = self.NET.predict(embedded_t),self.NET.predict_classes(embedded_t)
        print(prob,self.matcher[p_class[0]],to_predict)


P = Predictor()
P.tell('lock the door')
P.tell('close the door')
P.tell('open the door')
P.tell('open the tap')
P.tell('fetch the water')
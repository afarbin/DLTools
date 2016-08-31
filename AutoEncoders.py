#
# AutoEncoders.py
# Amir Farbin

from ModelWrapper import *

from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model

class DenseAutoEncoder(ModelWrapper):
    def __init__(self, Name, 
                 InputShape= (100,), 
                 Widths=[10],
                 EncodeActivation="relu",
                 DecodeActivation="sigmoid",
                 Loss="adadelta",
                 Optimizer="binary_crossentropy"):
        super(LSTMAutoEncoder,self).__init__(Name,Loss,Optimizer)

        self.InputShape=self.MetaData["InputShape"]=InputShape
        self.Widths=self.MetaData["Widths"]=Widths
        self.EncodeActivation=self.MetaData["EncodeActivation"]=EncodeActivation
        self.DecodeActivation=self.MetaData["DecodeActivation"]=DecodeActivation


    def Build():        
        # Input
        myInput = Input(shape=self.InputShape)
        myModel = myInput

        # Encode
        for i in range(0,len(Widths)):
            myModel = Dense(Widths[i], activation=self.EncodeActivation)(myModel)

        # Decode
        for i in range(len(Widths)-1,-1, 0):
            myModel = Dense(Widths[i], activation=self.EncodeActivation)(myModel)
        myModel = Dense(Widths[0], activation=self.DecodeActivation)(myModel)

        self.Model = Model(input=myInput, output=myModel)



class LSTMAutoEncoder(ModelWrapper):
    def __init__(self, Name, 
                 InputShape= (10,100), 
                 Widths=[10],
                 EncodeActivation="tanh",
                 DecodeActivation="tanh",
                 Loss="mse",
                 Optimizer="binary_crossentropy"):
        super(LSTMAutoEncoder,self).__init__(Name,Loss,Optimizer)

        self.InputShape=self.MetaData["InputShape"]=InputShape
        self.Widths=self.MetaData["Widths"]=Widths
        self.EncodeActivation=self.MetaData["EncodeActivation"]=EncodeActivation
        self.DecodeActivation=self.MetaData["DecodeActivation"]=DecodeActivation

    def Build(self):        
        # Input
        print self.InputShape
        myInput = Input(shape=self.InputShape)
        myModel = myInput

        # Encode
        for i in range(0,len(self.Widths)):
            print "Adding Encoder",i,self.Widths[i]
            myModel = LSTM(self.Widths[i],consume_less="gpu",
                           activation=self.EncodeActivation,
                           return_sequences=True)(myModel)
#            myModel = RepeatVector(self.InputShape[0])(myModel)

        # Decode
        for i in range(len(self.Widths)-1,-1, -1):
            myModel = LSTM(self.Widths[i],consume_less="gpu",
                           activation=self.DecodeActivation,
                           return_sequences=True)(myModel)

        myModel = LSTM(self.InputShape[1],consume_less="gpu",return_sequences=True)(myModel)

        self.Model = Model(input=myInput, output=myModel)

class LSTMAutoEncoder2(ModelWrapper):
    def __init__(self, Name, 
                 InputShape= (10,100), 
                 Widths=[10],
                 EncodeActivation="relu",
                 DecodeActivation="sigmoid",
                 Loss="adadelta",
                 Optimizer="binary_crossentropy"):
        super(LSTMAutoEncoder2,self).__init__(Name,Loss,Optimizer)

        self.InputShape=self.MetaData["InputShape"]=InputShape
        self.Widths=self.MetaData["Widths"]=Widths
        self.EncodeActivation=self.MetaData["EncodeActivation"]=EncodeActivation
        self.DecodeActivation=self.MetaData["DecodeActivation"]=DecodeActivation

    def Build(self):        
        # Input
        print self.InputShape
        myInput = Input(shape=self.InputShape)
        encoder = myInput

        # Encode
        for i in range(0,len(self.Widths)):
            print "Adding Encoder",i,self.Widths[i]
            encoder = LSTM(self.Widths[i],return_sequences=True)(encoder)

            ##        myModel = RepeatVector(self.InputShape[0])(myModel)

        # Decode
        decoder = encoder
        for i in range(len(self.Widths)-1,0, -1):
            decoder = LSTM(self.Widths[i],return_sequences=True)(decoder)

        # Reconstruct output
        for i in range(len(self.Widths)-1,-1, -1):
            myModel = LSTM(self.InputShape[1],return_sequences=True)(myModel)

        self.Model = Model(input=myInput, output=myModel)
        


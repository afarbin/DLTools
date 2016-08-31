#
# ModelWrapper.py
# Amir Farbin

import sys,os
import cPickle as pickle
from keras.models import model_from_json

class ModelWrapper(object):

    def __init__(self, Name, Loss=None, Optimizer=None):
        self.Name=Name
        self.Loss=Loss
        self.Optimizer=Optimizer
        self.MetaData={ "Name": Name, "Optimizer":Optimizer, "Loss":Loss }
        self.Initialize()

    def Initialize(self, Overwrite=False):
        try:
            os.mkdir("TrainedModels")
        except:
            pass

        self.OutDir="TrainedModels/"+self.Name
        self.InDir=self.OutDir

        if not Overwrite:
            i=1
            OutDir=self.OutDir
            while os.path.exists(OutDir):
                OutDir=self.OutDir+"."+str(i)
                i+=1

            self.OutDir=OutDir

        self.MetaData["OutDir"]=self.OutDir
                        
    def Save(self,OutDir=False):
        if OutDir:
            self.OutDir=OutDir

        try:
            os.makedirs(self.OutDir)
        except:
            print "Error making output Directory"

        open(self.OutDir+"/Model.json", "w").write( self.Model.to_json() )
        self.Model.save_weights(self.OutDir+"/Weights.h5",overwrite=True)
        pickle.dump(self.MetaData, open(self.OutDir+"/MetaData.pickle","wb"))

    def Load(self,InDir=False,MetaDataOnly=False,Overwrite=False):

        if InDir:
            self.InDir = InDir

        if not MetaDataOnly:
            self.Model = model_from_json( open(self.InDir+"/Model.json", "r").read() )
            self.Model.load_weights(self.InDir+"/Weights.h5")

        MetaData=pickle.load( open(self.InDir+"/MetaData.pickle","rb"))
        self.MetaData.update(MetaData)
        self.MetaData["InputMetaData"]=[MetaData]
        self.MetaData["InputDir"]=self.InDir

        NoneType=type(None)

        if "Optimizer" in self.MetaData.keys():
            self.Optimizer=self.MetaData["Optimizer"]

        if type(self.Optimizer)==NoneType:
            self.Optimizer="sgd"

        if "Loss" in self.MetaData.keys():
            self.Loss=self.MetaData["Loss"]

        if type(self.Loss)==NoneType:
            self.Loss="mse"

        self.Initialize(Overwrite=Overwrite)

    def Compile(self, Loss=False, Optimizer=False):
        if Loss:
            self.Loss=Loss
        if Optimizer:
            self.Optimizer=Optimizer
        self.Model.compile(loss=self.Loss, optimizer=self.Optimizer)
    
    def Train(self, X, y, Epochs, BatchSize, Callbacks=None,  validation_split=0.):
        History=self.Model.fit(X, y, nb_epoch=Epochs, batch_size=BatchSize,
                               callbacks=Callbacks, validation_split= validation_split)
        self.History=History
        self.MetaData["History"]=History.history

    def Build(self):
        pass



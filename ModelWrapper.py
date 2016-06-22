#
# ModelWrapper.py
# Amir Farbin

import sys,os
import cPickle as pickle

class ModelWrapper(object):
    def __init__(self, Name, Loss=None, Optimizer=None):
        self.Name=Name
        self.Loss=Loss
        self.Optimizer=Optimizer
        self.MetaData={ "Name": Name }
        self.Initialize()

    def Initialize(self, Overwrite=False):
        try:
            os.mkdir("TrainedModels")
        except:
            pass

        self.OutDir="TrainedModels/"+self.Name
        
        if Overwrite:
            i=1
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

    def Load(self,OutDir=False,MetaDataOnly=False,Overwrite=False):

        if OutDir:
            self.OutDir = OutDir

        if not MetaDataOnly:
            self.Model = model_from_json( open(self.OutDir+"/Model.json", "r").read() )
            self.Model.load_weights(self.OutDir+"/Weights.h5")

        MetaData=pickle.load( open(self.OutDir+"/MetaData.pickle","rb"))
        self.MetaData.update(MetaData)
        self.MetaData["InputMetaData"]=[MetaData]
        self.MetaData["InputDir"]=self.OutDir

        self.Initialize(Overwrite=Overwrite)

    def Compile(self, Loss=False, Optimizer=False):
        if Loss:
            self.Loss=Loss
        if Optimizer:
            self.Optimizer=Optimizer
        self.Model.compile(loss=self.Loss, optimizer=self.Optimizer)
    
    def Train(self, X, y, Epochs, BatchSize):
        History=self.Model.fit(X, y, nb_epoch=Epochs, batch_size=BatchSize)
        self.History=History
        self.MetaData["History"]=History.history

    def Build(self):
        pass



#python -i AnomalyDetectTestMultiModel.py -R "TrainedModels/*" -D "DataCache/Pattern_1000000.0_10_100_10_[3,10]_2_(5,10)_[1,5]_0.05_(5,15).h5"

from scipy import misc as m
import h5py
from ROOT import *
from root_numpy import fill_hist
import os
import numpy as np

def Analyze(X,MyModel,basename="Sequence",directory=".",makepng=True,makerootpng=False):
    try:
        os.makedirs(directory)
    except:
        pass

    Result_Y=MyModel.Model.predict(X)

    filenamebase=directory+"/"+basename
        
    tf=TFile(filenamebase+".root","RECREATE")
    c1=TCanvas("c1")

    Xshape=X.shape

    for i in xrange(0,Xshape[0]):
        seq=X[i]
        seqY=Result_Y[i]
        name=filenamebase+"_"+str(i)

        if makepng:
            m.imsave(name+"_X.png",seq)
            m.imsave(name+"_Y.png",seqY)
            m.imsave(name+"_R.png",seq-seqY)
        
        shape=seq.shape

        histX=TH2F(name+"_X",name+"_X",shape[0],0,100,shape[1],0,10)
        histY=TH2F(name+"_Y",name+"_Y",shape[0],0,100,shape[1],0,10)
        histR=TH2F(name+"_R",name+"_R",shape[0],0,100,shape[1],0,10)

        for i in xrange(1,shape[0]+1):
            for j in xrange(1,shape[1]+1):
                histX.SetBinContent(i,j,seq[i-1][j-1])
                histY.SetBinContent(i,j,seqY[i-1][j-1])

        histR.Add(histX)
        histR.Add(histY,-1)

        histX.Write()
        histY.Write()
        histR.Write()

        if makerootpng:
            histX.Draw()
            c1.Print(name+"_X.hist.png")

            histY.Draw()
            c1.Print(name+"_Y.hist.png")

            histR.Draw()
            c1.Print(name+"_R.hist.png")

    tf.Close()

from scipy import stats
from array import array

def AnalyzeInjection(Xs,MyModel,basename,directory=".",MakeRootHists=False):
    try:
        os.makedirs(directory)
    except:
        pass

    filenamebase=directory+"/"+basename

    tf=TFile(filenamebase+".root","RECREATE")
    t = TTree( 'RecoError', 'Tree' )

    if MakeRootHists:
        c1=TCanvas("c1")
    
        #    NThreshold=11
        #    MaxThreshold=0.1
        #    NThreshold_b = array( 'i', [ int( NThreshold) ] )
        #    t.Branch( 'NThreshold', NThreshold_b, 'NThreshold/I' )

    N_Samples=Xs[0].shape[1]
    N_Samples_b = array( 'i', [ int( N_Samples ) ] )
    t.Branch( 'N_Samples', N_Samples_b, 'N_Samples/I' )

    RecoError_b = [] 
    RecoErrorT_b  = [] 
    for i in xrange(0,len(Xs)):
        RecoError_b.append( array( 'f', N_Samples*[ 0. ] ))
        t.Branch( 'RecoError_'+str(i), RecoError_b[i], 'RecoError[N_Samples]_'+str(i)+'/F' )

    Xshape=Xs[0].shape

    print Xshape

    Result_Y=[]
    for X in Xs:
        Result_Y.append(MyModel.Model.predict(X))

    for i in xrange(0,Xshape[0]):

        name= filenamebase+"_"+str(i)
        Image=np.zeros( (Xshape[1] * 3 , Xshape[2] * len(Xs)))
        for j in xrange(0,len(Xs)):
            X=Xs[j][i]
            Y=Result_Y[j][i] 
            R=X-Y

            Image[0 * Xshape[1] : 1* Xshape[1],j * Xshape[2] : (j+1) * Xshape[2]] = X
            Image[1 * Xshape[1] : 2* Xshape[1],j * Xshape[2] : (j+1) * Xshape[2]] = Y
            Image[2 * Xshape[1] : 3* Xshape[1],j * Xshape[2] : (j+1) * Xshape[2]] = (np.max(X)/np.max(R))*R

            R_Local=R/X
            R_Local_sq=R_Local**2
            R_Local_sq_Avg=np.average(R_Local_sq,axis=1)

            for iii in xrange(0,N_Samples):
                RecoError_b[j][iii]=R_Local_sq_Avg[iii]

## Misguieded attempt to pull out anomally by looking at sum reco err as function of threshold
#            RecoError_b[j][0]= np.sum(R_Local_sq)
#
#            iii=1
#            for thres in np.arange(0,MaxThreshold,MaxThreshold/(NThreshold-1)):
#                RecoError_b[j][iii]=np.sum(stats.threshold(R, threshmin=thres, newval=0)**2)
#
#                iii+=1

            if MakeRootHists:
                nameh=name+"_"+str(j)
                histX=TH2F(nameh+"_X",nameh+"_X",shape[0],0,100,shape[1],0,10)
                histY=TH2F(nameh+"_Y",nameh+"_Y",shape[0],0,100,shape[1],0,10)
                histR=TH2F(nameh+"_R",nameh+"_R",shape[0],0,100,shape[1],0,10)

                for ii in xrange(1,shape[0]+1):
                    for jj in xrange(1,shape[1]+1):
                        histX.SetBinContent(ii,jj,X[ii-1][jj-1])
                        histY.SetBinContent(ii,jj,Y[ii-1][jj-1])
                        histY.SetBinContent(ii,jj,R[ii-1][jj-1])

                histX.Write()
                histY.Write()
                histR.Write()

        t.Fill()
        m.imsave(name+"_"+str(i)+".png",Image)
    
    t.Write()

    hS=TH1F("hS","RecoErrorS",100,-10,0)
    t.Draw("log(RecoError_2)>>hS","RecoError_1<1000000")

    hB=TH1F("hB","RecoErrorB",100,-10,0)
    t.Draw("log(RecoError_2)>>hB","RecoError_1>1000000")

    hB.SetLineColor(2)
    c1=TCanvas("c1")
    
    hB.Draw()
    hS.Draw("same")
    
    c1.Print(filenamebase+"_RecoError.png")

    tf.Close()

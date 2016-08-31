
from scipy import misc as m
import h5py
from ROOT import *
from root_numpy import fill_hist

f=h5py.File("DataCache/Pattern_1000000.0_10_100_10_[3,10]_2_(5,10)_[1,5]_0.05_(5,15).h5")

tf=TFile("Images/Sequences.root","RECREATE")

c1=TCanvas("c1")

for i in xrange(0,100):
    seq=f["CachedData"]["Sequence"][i][0]
    name="Sequence_"+str(i)
    m.imsave("Images/"+name+".png",seq)
    shape=seq.shape
    hist=TH2F(name,name,shape[0],0,100,shape[1],0,10)
    for i in xrange(1,shape[0]+1):
        for j in xrange(1,shape[1]+1):
            hist.SetBinContent(i,j,seq[i-1][j-1])

    hist.Write()
    hist.Draw()

    c1.Print("Images/"+name+".pdf")

f.close()
tf.Close()

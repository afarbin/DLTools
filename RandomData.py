import numpy as np
import os
from tables import *
import h5py


def RandomSequenceData(N_Examples, N_Inputs, N_Samples, FractionTest=.1):
    
    N_Test=int(round(FractionTest*N_Examples))
    N_Train=N_Examples-N_Test

    return ( np.random.rand(N_Train, N_Inputs, N_Samples),
             np.random.rand(N_Test, N_Inputs, N_Samples))


def RandomSequenceGenerator(batchsize, N_Inputs, N_Samples):
    while True:
        X=np.random.rand(batchsize, N_Inputs, N_Samples)
        yield (X,X)

def PatternGenerator(batchsize, N_Inputs, N_Samples,
                     N_Patterns=10, PatternSamples=5, NoiseSigma=0, 
                     A_range=1, f_range=1, s_range=.05, L_range=10, 
                     verbose=False):

    WG=WindowGenerator(N_Inputs=N_Inputs, N_Samples=N_Samples, 
                       N_Patterns=N_Patterns,
                       PatternSamples=PatternSamples, NoiseSigma=NoiseSigma, 
                       A_range=A_range,f_range=f_range,s_range=s_range,L_range=L_range)
    
    X=np.zeros((batchsize, N_Inputs, N_Samples))
    count=0

    while True:
        for i in xrange(0,batchsize):
            count+=1
            if verbose: 
                if count%1000==0 : print count
            X[i]=WG.GenerateOne()

        yield (X,X)


def SplitTrainTest(X, FractionTest):
    N_Examples=int(np.shape(X)[0])
    N_Test=int(round(FractionTest*N_Examples))
    N_Train=int(N_Examples-N_Test)

    Train_X=X[0:N_Train]
    Test_X=X[N_Train:N_Train+N_Test]
    return (Train_X, Test_X)



def GeneratePatternSample( N_Examples=0, N_Inputs=0, N_Samples=0, FractionTest=0,
                           N_Patterns=10, PatternSamples=5, NoiseSigma=0, 
                           A_range=1,f_range=1,s_range=.05,L_range=10, cache=True,
                           verbose=True,filename="",MaxLoad=-1):
    if cache:
        if filename=="":
            name="Pattern"
            for n in [N_Examples, N_Inputs, N_Samples,
                      N_Patterns, PatternSamples, NoiseSigma,
                      A_range,f_range,s_range,L_range]:
                name+= "_" + str(n).replace(" ","")
        
            filename="DataCache/"+name+".h5"
       
        try:
            os.mkdir("DataCache")
        except:
            print "CacheDirectory Exists."

        if os.path.isfile(filename): 
            print "Loading Data From ", filename
            f=h5py.File(filename)
            if MaxLoad>0:
                X=f["CachedData"]["Sequence"]["Sequence"][:int(MaxLoad)]
            else:
                X=f["CachedData"]["Sequence"]["Sequence"]
            return SplitTrainTest(X,FractionTest) 
        
        class SequenceExample(IsDescription):
            Sequence=Float32Col(shape=( N_Samples,N_Inputs))

        h5file = open_file(filename, mode="w")
        print "Writing out to:",filename
        FILTERS = Filters(complib='zlib', complevel=5)
        group = h5file.create_group("/", 'CachedData', 'Cached Data')
        table = h5file.create_table(group, 'Sequence', SequenceExample, 
                                    "a sequence",filters=FILTERS)
        truth_I = h5file.create_vlarray(group,"Pattern_I",Int16Atom(shape=()),"Index",filters=Filters(1))

        truth_A = h5file.create_vlarray(group,"Pattern_A",Float32Atom(shape=()),"Pattern Index",filters=Filters(1))
        truth_L = h5file.create_vlarray(group,"Pattern_L",Int16Atom(shape=()),"Length",filters=Filters(1))
        truth_i = h5file.create_vlarray(group,"Pattern_i",Int16Atom(shape=()),"Location in Window",filters=Filters(1))
        truth_s = h5file.create_vlarray(group,"Pattern_s",Float32Atom(shape=()),"Noise",filters=Filters(1))
        truth_f = h5file.create_vlarray(group,"Pattern_f",Float32Atom(shape=()),"Frequency",filters=Filters(1))


    WG=WindowGenerator(N_Inputs=N_Inputs, N_Samples=N_Samples, 
                       N_Patterns=N_Patterns,
                       PatternSamples=PatternSamples, NoiseSigma=NoiseSigma, 
                       A_range=A_range,f_range=f_range,s_range=s_range,L_range=L_range)


    X=np.zeros((N_Examples, N_Samples, N_Inputs))

    for i in xrange(0,int(N_Examples)):
        if verbose:
            if i%1000==0:
                print i
        X0,T=WG.GenerateOne(True)
        X[i]=X0

        if cache:
            anExample=table.row
            anExample["Sequence"]=X0
            anExample.append()

            # Store truth
            I=[]
            A=[]
            L=[]
            i_W=[]
            s=[]
            f=[]

            for t in T:
                I.append(t[0])
                A.append(t[1]["A"])
                L.append(t[1]["L"])
                i_W.append(t[1]["i_W"])
                s.append(t[1]["s"])
                f.append(t[1]["f"])

            truth_I.append(I)
            truth_A.append(A)
            truth_L.append(L)
            truth_i.append(i_W)
            truth_s.append(s)
            truth_f.append(f)
            
    
    if cache:

        h5TruthFile=h5py.File("DataCache/"+name+".truth.h5","w") 
        for i in xrange(0,len(WG.Patterns)):
            Truth=h5TruthFile.create_dataset("Pattern"+str(i),data=WG.Patterns[i].ThePattern)

        h5TruthFile.close()
        h5file.close()

    return SplitTrainTest(X,FractionTest) 

# Create N Classes of random events.
# 


class Pattern(object):
    def __init__(self, N_Inputs, N_Samples, A, f, s, L, N_Draws=1):
        self.N_Inputs=int(N_Inputs)
        self.N_Samples=int(N_Samples)
        self.N_Draws=N_Draws
        self.A=A
        self.f=f
        self.s=s
        self.L=L

        self.GenerateTemplate()

    def GenerateTemplate(self):
        self.ThePattern=np.random.rand(self.N_Inputs,self.N_Samples)

    def GenerateParam(self,x):
        if type(x) == list or type(x) == tuple :
            I=x[1]-x[0]
            return I*np.random.random()+x[0]
        else:
            return x


    def Generate(self,WindowSize, Truth=False):
        # Amplitude
        A=self.GenerateParam(self.A)
       
        # Length
        L=int(self.GenerateParam(self.L))
        
        # Build the Signal
        TheSignal=np.zeros((L,self.N_Inputs))

        for i in xrange(0,L):
            ii= int(self.N_Samples * float(i)/float(L))
            
            for j in xrange(0,self.N_Inputs):
                TheSignal[i][j]=np.random.normal(A*self.ThePattern[j][ii],self.s)
        
        # Start Location within window

        i_W = int((WindowSize-L) * np.random.random())

        
        out= np.pad(TheSignal, 
                    ((i_W, abs(WindowSize-L-i_W)),(0,0)), 
                    "constant", constant_values=0)        

        if Truth:
            return out, {"A":A,"L":L,"i_W":i_W, 
                         "s":self.s,
                         "f":self.f}
        else:
            return out


# A = Amplitude
# s = noise sigma
# L = length of pattern
# t = ???

class PatternGenerator(object):
    def __init__(self,N_Inputs, N_Samples, N_Patterns,
                 A_range=[2,10],f_range=[1,10],s_range=[1,5],L_range=[1,5]):
        self.N_Inputs=N_Inputs
        self.N_Samples=N_Samples
        self.N_Patterns=N_Patterns
        
        self.A_range=A_range
        self.f_range=f_range
        self.s_range=s_range
        self.L_range=L_range

        self.GenerateParameters()

    def Flat(self,N, range=[0.,1.]):
        if type(range) == list:
            I=range[1]-range[0]
            return I*np.random.rand(N)+range[0]
        else:
            if type(range) == tuple:
                return [range]*N
        
        return range*np.ones(N)


    def GenerateParameters(self):
            self.PatternSample_N=self.Flat(self.N_Patterns,self.N_Samples)
            self.A=self.Flat(self.N_Patterns,self.A_range)
            self.f=self.Flat(self.N_Patterns,self.f_range)
            self.s=self.Flat(self.N_Patterns,self.s_range)
            self.L=self.Flat(self.N_Patterns,self.L_range)

    def Generate(self):
        self.Patterns=[]
        for i in xrange(0,self.N_Patterns):
            self.Patterns.append(Pattern(self.N_Inputs,
                                         self.PatternSample_N[i],
                                         self.A[i],
                                         self.f[i],
                                         self.s[i],
                                         self.L[i]))

        return self.Patterns
                 
class WindowGenerator(object):
    def __init__(self, N_Inputs, N_Samples, N_Patterns, PatternSamples, NoiseSigma, 
                 A_range=[2,10],f_range=[1,10],s_range=[1,5],L_range=[1,5]):
        
        
        self.N_Inputs=N_Inputs
        self.N_Samples=N_Samples
        self.N_Patterns=N_Patterns
        self.NoiseSigma=NoiseSigma
        self.A_range=A_range
        self.f_range=f_range
        self.s_range=s_range
        self.L_range=L_range
        self.PatternSamples=PatternSamples

        PG=PatternGenerator(N_Inputs,
                            PatternSamples,
                            N_Patterns,
                            A_range,
                            f_range,
                            s_range,
                            L_range)

        self.Patterns=PG.Generate()


        self.FrequencyVector=(N_Patterns+1)*[0]

        sum=0.
        # Normalize Pattern Frequency        
        for i in xrange(0,N_Patterns):
            sum+=self.Patterns[i].f
            self.FrequencyVector[i+1]+=sum

        self.PatternsPerWindow=sum
        self.FrequencyVector=np.array(self.FrequencyVector)/sum

    # Generate Patterns
    def GenerateOne(self, Truth=False):

        #Create Window with noise
        Window=self.NoiseSigma*np.random.randn( self.N_Samples, self.N_Inputs)

        #Draw number of patterns
        N_P=int(np.random.poisson(self.PatternsPerWindow))
        if N_P==0:
            N_P=1

        ChoosenPatterns=[]
        #Generate Patterns
        for i in xrange(0,N_P):
            # Pick a pattern
            x=np.random.random()
            i_P=np.digitize(x,self.FrequencyVector)-1

            P,PatternTruth=self.Patterns[i_P].Generate(self.N_Samples,True)
            ChoosenPatterns.append((i_P,PatternTruth))
            
            Window+=P

        if Truth:
            return Window,  ChoosenPatterns
        else:
            return Window


def TestRandomData():

    #TestP=Pattern( N_Inputs=10, N_Samples=5, A=10, f=10, s=.1, L=10, N_Draws=1)

    TestP=Pattern( N_Inputs=3, N_Samples=3, A=100, f=1, s=.11, L=3, N_Draws=1)


    #TestWG=WindowGenerator(N_Inputs=5, N_Samples=30, N_Patterns=1, PatternSamples=5, NoiseSigma=0, 
    #                       A_range=[1,2],f_range=[1,10],s_range=[1,5],L_range=[3,10])

    TestWG=WindowGenerator(N_Inputs=5, N_Samples=30, N_Patterns=1, PatternSamples=5, NoiseSigma=0, 
                           A_range=1,f_range=1,s_range=.05,L_range=10)


    W,CP= TestWG.GenerateOne(True)

    for p in xrange(0,len(CP)):
        print "Pattern: " 

        print TestWG.Patterns[CP[p][0]].ThePattern[0]
        print CP[p]
        i=CP[p][1]["i_W"]
        L=CP[p][1]["L"]
        print "Generated: ", i, L
        
        print W[0][i:i+L]
    



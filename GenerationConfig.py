import random

from Permutator import Permutator

Name="LSTMAutoEncoder"

Config={
    "Epochs":0,
    "BatchSize":2048,
    
    "LearningRate":0.005,

    "Decay":0.,
    "Momentum":0.,
    "Nesterov":0.,

    "N_Samples":100, # Samples in a window

    "WeightInitialization":"'normal'",
}

# For Random Generation

Config["N_Examples"]=1e6
Config["FractionTest"]=0.1
Config["N_Inputs"]=10

# Pattern Definition
# Fixed Parameters
Config["NoiseSigma"]= 2     # Noise baseline
Config["N_Patterns"]=10     # Number of possible random patterns classes

# Variable (random) Parameters
# Convention : 
#    X = 5       ==> X is always 5
#    X = [1,10]  ==> A given pattern will have X between 1 and 5
#    X = (1,10)  ==> An instance of a given pattern will have x between 1 and 5 

Config["PatternSamples"]=[3,10]  # Inherent pattern length
Config["L_range"]=(5,15)        # Pattern Length
Config["A_range"]=(5,10)         # Amplitude Range
Config["f_range"]=[1,5]         # Frequency Range ( f * N_Patterns = mean PatternsPerWindow )
Config["s_range"]=0.05           # Additional Noise for pattern

# Network Architecture
Params={ "Widths": [1*[16],1*[32],1*[64],1*[128], 1*[256],
                    2*[16],2*[32],2*[64],2*[128], 2*[256],
                    3*[16],3*[32],3*[64],3*[128], 3*[256],
                    4*[16],4*[32],4*[64],4*[128], 4*[256],
                    5*[16],5*[32],5*[64],5*[128], 5*[256]],

         "EncodeActivation": ["'sigmoid'","'relu'"],
         "DecodeActivation": ["'sigmoid'","'softmax'"],
         "Optimizer": ["'adam'"],
         "Loss": ["'mean_squared_error'","'adadelta'"  ],

         }


PS=Permutator(Params)
Combos=PS.Permutations()

print "HyperParameter Scan: ", len(Combos), "possible combiniations."

if "HyperParamSet" in dir():
    i=int(HyperParamSet)
else:
    # Set Seed based on time
    random.seed()
    i=int(round(len(Combos)*random.random()))
    print "Randomly picking HyperParameter Set"

print "Picked combination: ",i

for k in Combos[i]:
    Config[k]=Combos[i][k]

for MetaData in Params.keys():
    val=str(Config[MetaData]).replace('"',"")
    val=val.replace("'","")
    Name+="_"+val.replace(" ","")

print "Model Filename: ",Name




import random

from Permutator import Permutator

Name="LSTMAutoEncoder-Default"

Config={
    "Epochs":10,
    "BatchSize":128,
    
    "LearningRate":0.005,

    "Decay":0.,
    "Momentum":0.,
    "Nesterov":0.,

    "N_Samples":100, # Samples in a window

    "WeightInitialization":"'normal'",

    "Widths":[16],

    "EncodeActivation": "'tanh'",
    "DecodeActivation": "'tanh'",
    "Optimizer": "'rmsprop'",
    "Loss": "'mean_squared_error'",

}

# For Random Generation

Config["N_Examples"]=1e5
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


print "Model Filename: ",Name




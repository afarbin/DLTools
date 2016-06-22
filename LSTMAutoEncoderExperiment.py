import sys,os,argparse

# Configuration of this job
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',default="LSTMAutoEncoderDefaultScanConfig.py")

parser.add_argument('-L', '--LoadModel',default=False)
parser.add_argument('--gpu', dest='gpuid', default="")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--NoTrain', action="store_true")
parser.add_argument('-s',"--hyperparamset", default="0")

parser.add_argument('--generator', action="store_true")


args = parser.parse_args()
Train = not args.NoTrain
UseGPU = not args.cpu
gpuid = args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)
ConfigFile = args.config
useGenerator = args.generator

# Configuration from PBS:
if "PBS_ARRAYID" in os.environ:
    HyperParamSet = int(os.environ["PBS_ARRAYID"])

if "PBS_QUEUE" in os.environ:
    if "cpu" in os.environ["PBS_QUEUE"]:
        UseGPU=False
    if "gpu" in os.environ["PBS_QUEUE"]:
        UseGPU=True
        gpuid=int(os.environ["PBS_QUEUE"][3:4])

if UseGPU:
    print "Using GPU",gpuid
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%s,floatX=float32,force_device=True" % (gpuid)
else:
    print "Using CPU."

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Load the Data
from RandomData import *

if useGenerator:
    myGenerator = PatternGenerator(BatchSize,N_Inputs,N_Samples,
                                   N_Patterns=N_Patterns,
                                   PatternSamples=PatternSamples,
                                   NoiseSigma=NoiseSigma,
                                   A_range=A_range,
                                   f_range=f_range,
                                   s_range=s_range,
                                   L_range=L_range  )
else:
    (Train_X, Test_X) = GeneratePatternSample(N_Examples,N_Inputs,N_Samples,FractionTest,
                                              N_Patterns=N_Patterns,
                                              PatternSamples=PatternSamples,
                                              NoiseSigma=NoiseSigma,
                                              A_range=A_range,
                                              f_range=f_range,
                                              s_range=s_range,
                                              L_range=L_range)


# Build the Model
from AutoEncoders import LSTMAutoEncoder

# Instantiate a LSTM AutoEncoder
MyModel=LSTMAutoEncoder(Name, 
                        InputShape=(N_Samples,N_Inputs),
                        Widths=Widths,
                        EncodeActivation=EncodeActivation,
                        DecodeActivation=DecodeActivation,
                        Loss=Loss,
                        Optimizer=Optimizer)

# Build it
MyModel.Build()

# Print out the Model Summary
MyModel.Model.summary()

# Compile The Model
print "Compiling Model."
MyModel.Compile() 

# Train
if Train:
    print "Training."

    if useGenerator:
        MyModel.Model.fit_generator(myGenerator, samples_per_epoch = N_Examples, 
                                    nb_epoch = Epochs, 
                                    verbose=2,  callbacks=[], 
                                    validation_data=None, class_weight=None)
    
        # Evaluate Score on Test sample
        score = MyModel.Model.evaluate(Test_X, Test_X)

    else:
        MyModel.Train(Train_X, Train_X, Epochs, BatchSize)
    
        # Evaluate Score on Test sample
        score = MyModel.Model.evaluate(Test_X, Test_X)

    print "Final Score:", score

# Analysis

# Save Model
MyModel.Save()



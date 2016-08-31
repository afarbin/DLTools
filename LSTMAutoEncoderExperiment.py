import sys,os,argparse


# Configuration of this job
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',default="LSTMAutoEncoderDefaultScanConfig.py")

parser.add_argument('-L', '--LoadModel',default=False)
parser.add_argument('--gpu', dest='gpuid', default="")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--NoTrain', action="store_true")
parser.add_argument('--NoAnalysis', action="store_true")
parser.add_argument('--Test', action="store_true")
parser.add_argument('-s',"--hyperparamset", default="0")
parser.add_argument('--generator', action="store_true")

# Configure based on commandline flags... this really needs to be cleaned up
args = parser.parse_args()
Train = not args.NoTrain
Analyze = not args.NoAnalysis
TestMode = not args.Test
UseGPU = not args.cpu
gpuid = args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)
ConfigFile = args.config
useGenerator = args.generator

LoadModel=args.LoadModel

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

from keras.callbacks import EarlyStopping

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

# Normalize the Data... seems to be critical!
Norm=np.max(Train_X)
Train_X=Train_X/Norm
Test_X=Test_X/Norm

# Build/Load the Model
from AutoEncoders import LSTMAutoEncoder
from ModelWrapper import ModelWrapper

# Instantiate a LSTM AutoEncoder... 

if LoadModel:
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/":
        LoadModel=LoadModel[:-1]
    Name=os.path.basename(LoadModel)
    MyModel=ModelWrapper(Name)
    MyModel.InDir=os.path.dirname(LoadModel)
    MyModel.Load()
else:
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

    callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min') ]

    if useGenerator:
        MyModel.Model.fit_generator(myGenerator, samples_per_epoch = N_Examples, 
                                    nb_epoch = Epochs, 
                                    verbose=2,  
                                    validation_data=None, class_weight=None,
                                    callbacks=callbacks)
    
        # Evaluate Score on Test sample
        score = MyModel.Model.evaluate(Test_X, Test_X)

    else:
        MyModel.Train(Train_X, Train_X, Epochs, BatchSize,  validation_split=0.1, Callbacks=callbacks)
    
        # Evaluate Score on Test sample
        score = MyModel.Model.evaluate(Test_X, Test_X)

    print "Final Score:", score

# Analysis
if Analyze:
    import AutoEncoderAnalysis
    N_Analyze=10
    AutoEncoderAnalysis.Analyze(Test_X[0:N_Analyze],MyModel,directory=MyModel.OutDir+"/Analysis",makepng=True)

# Save Model
if Train:
    MyModel.Save()



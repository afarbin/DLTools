#python -i AnomalyDetectTestMultiModel.py -R "TrainedModels/*" -D "DataCache/Pattern_1000000.0_10_100_10_[3,10]_2_(5,10)_[1,5]_0.05_(5,15).h5"


import sys,os,argparse
import glob

# Configuration of this job
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',default="AnomalyDetectTestConfig.py")

parser.add_argument('-L', '--LoadModel',default=False)
parser.add_argument('-D', '--LoadData',default=False)
parser.add_argument('--gpu', dest='gpuid', default="")
parser.add_argument('--N_Inject', dest='N_Analyze', default="10")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--NoAnalysis', action="store_true")
parser.add_argument('--Test', action="store_true")
parser.add_argument('-s',"--hyperparamset", default="0")
parser.add_argument('--generator', action="store_true")

parser.add_argument('-R', '--LoadDirectory',default=False)

# Configure based on commandline flags... this really needs to be cleaned up
args = parser.parse_args()
Analyze = not args.NoAnalysis
TestMode = not args.Test
UseGPU = not args.cpu
gpuid = args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)
ConfigFile = args.config
useGenerator = args.generator
Train=False
LoadModel=args.LoadModel
LoadData=args.LoadData

LoadDirectory=args.LoadDirectory

ModelDirs=glob.glob(LoadDirectory)


N_Analyze=int(args.N_Analyze)

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

# Load the original sample
(Train_X, Test_X) = GeneratePatternSample(filename=LoadData,FractionTest=0.1,MaxLoad=N_Analyze/0.1)
N_Examples=min(N_Analyze,Test_X.shape[0])
N_Samples=Test_X.shape[1]
N_Inputs=Test_X.shape[2]

# Create New Patterns to Inject
(Inject_X, Inject_Test_X) = GeneratePatternSample(N_Examples,N_Inputs,N_Samples,FractionTest=0,
                                                  N_Patterns=N_Patterns,
                                                  PatternSamples=PatternSamples,
                                                  NoiseSigma=0., # No additional Noise
                                                  A_range=A_range,
                                                  f_range=f_range,
                                                  s_range=s_range,
                                                  L_range=L_range,
                                                  cache=False)

# Inject the new patterns
Injected_X=Test_X[:N_Examples]+Inject_X

# Normalize the Data... seems to be critical!
Norm=np.max(Train_X)  # Use the same normalization as the training sample
Train_X=Train_X/Norm
Test_X=Test_X/Norm
Inject_X=Inject_X/Norm
Injected_X=Injected_X/Norm

# Build/Load the Models
from ModelWrapper import ModelWrapper


for ModelDir in ModelDirs:
    try:
        print "Loading Model From:",ModelDir
        LoadModel=ModelDir
        if LoadModel[-1]=="/":
            LoadModel=LoadModel[:-1]

        Name=os.path.basename(LoadModel)
        MyModel=ModelWrapper(Name)
        MyModel.InDir=LoadModel
        MyModel.Load()

        # Print out the Model Summary
        MyModel.Model.summary()

        # Compile The Model
        print "Compiling Model."
        MyModel.Compile() 

        # Analysis
        import AutoEncoderAnalysis

 
        AutoEncoderAnalysis.AnalyzeInjection([ Test_X[0:N_Analyze], 
                                               Inject_X[0:N_Analyze], 
                                               Injected_X[0:N_Analyze]], 
                                             MyModel, 
                                            basename="Injection",
                                             directory=MyModel.OutDir+"/Analysis")

        print "Output to:",MyModel.OutDir

    except:
        print "Failed."

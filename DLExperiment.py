import sys,os,argparse

# Configuration of this job
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',default="DefaultConfig.py")

parser.add_argument('-L', '--LoadModel',default=False)
parser.add_argument('--gpu', dest='gpuid', default="")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--NoTrain', action="store_true")
parser.add_argument('-s',"--hyperparamset", default="0")

args = parser.parse_args()

UseGPU=not args.cpu
gpuid=args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)
ConfigFile=args.config

if UseGPU:
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%s,floatX=float32,force_device=True" % (gpuid)

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Load the Data
from RandomData import RandomSequenceData

(Train_X, Test_X) = RandomSequenceData(N_Examples,N_Inputs)

# Build the Model
from AutoEncoders import LSTMAutoEncoder

# Instantiate a LSTM AutoEncoder
MyModel=LSTMAutoEncoder(Name, 
                        InputShape=InputShape,
                        Widths=Widths,
                        EncodeActivation=EncodeActivation,
                        DecodeActivation=DecodeActivation,
                        Loss=Loss,
                        Optimizer=Optimizer)

# Print out the Model Summary
MyModel.model.summary()

# Compile The Model
print "Compiling Model."
MyModel.Compile(loss=loss
                optimizer=optimizer)

# Train
if Train:
    print "Training."
    MyModel.Train(X_train, X_train, Epochs, BatchSize)
    
    # Evaluate Score on Test sample
    score = MyModel.model.evaluate(X_test, X_test)
    print "Final Score:", score

# Analysis

# Save Model
MyModel.Save()



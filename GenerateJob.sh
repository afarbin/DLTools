
#PBS -V

cd /work/afarbin/code/DLTools
source setup.sh

mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output > $output

python LSTMAutoEncoderExperiment.py --NoTrain  -C GenerationConfig.py &> $output




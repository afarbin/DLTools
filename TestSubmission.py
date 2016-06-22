#PBS -l nodes=1:ppn=1,mem=1g,walltime=72:00:00
#PBS -q gpu0queue
#PBS -m abe
#PBS -V

output=out.log

pwd > $output
printenv >> $output


cd /work/afarbin/code/DLTools
source setup.sh
#python  LSTMAutoEncoderExperiment.py --gpu 2 -s 10 &>> $output

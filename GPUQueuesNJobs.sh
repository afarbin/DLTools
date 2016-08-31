
# Usage: sudo ./GPUQueuesNJobs.sh 2

echo Setting Jobs Per GPU queue to $1

qmgr -c "set queue gpu0queue max_running=$1"
qmgr -c "set queue gpu0queue max_queuable=$1"
qmgr -c "set queue gpu0queue max_user_run=$1"

qmgr -c "set queue gpu1queue max_running=$1"
qmgr -c "set queue gpu1queue max_queuable=$1"
qmgr -c "set queue gpu1queue max_user_run=$1"

qmgr -c "set queue gpu2queue max_running=$1"
qmgr -c "set queue gpu2queue max_queuable=$1"
qmgr -c "set queue gpu2queue max_user_run=$1"


qmgr -c "list queue gpuqueue"

qmgr -c "list queue gpu0queue"
qmgr -c "list queue gpu1queue"
qmgr -c "list queue gpu2queue"

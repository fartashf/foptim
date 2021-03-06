from __future__ import print_function
import math


def bolt(sargs, num_runs, job_args):
    """
    rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid G --run_name X
    pattern=""; for i in 1 2; do ./kill.sh $i $pattern; done
    ./start.sh
    """
    if len(job_args) > 0:
        jobs_0 = job_args[0]
    else:
        jobs_0 = ['bolt3_gpu0', 'bolt3_gpu1', 'bolt3_gpu2',
                  'bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2', 'bolt2_gpu3',
                  'bolt1_gpu3', 'bolt1_gpu2',
                  'bolt1_gpu0', 'bolt1_gpu1',
                  ]

    # Number of parallel jobs on each machine
    # validate start.sh
    if len(job_args) > 1:
        njobs = job_args[1]
    else:
        njobs = [3]*7 + [2]*2 + [0, 2]
    jobs = []
    for s, n in zip(jobs_0, njobs):
        jobs += ['%s_job%d' % (s, i) for i in range(n)]
    parallel = False  # each script runs in sequence
    print(num_runs)
    return jobs, parallel


def slurm(sargs, num_runs, job_args):
    """
    vector(q): gpu, wsgpu
    vaughan(vremote): p100, t4

    rm jobs/*.sh jobs/log/* -f && python grid_run.py --grid G --run_name X \
    --task_per_job 1 --job_limit 12 --partition p100,t4
    sbatch jobs/slurm.sbatch
    squeue -u <user>
    scancel -u <user>
    """
    njobs = int(math.ceil(num_runs/sargs.task_per_job))
    ntasks = sargs.job_limit
    partition = sargs.partition
    # njobs, ntasks, partition = sargs.split(',', 2)
    # njobs = int(njobs)
    # ntasks = int(ntasks)
    # njobs = 5  # Number of array jobs
    # ntasks = 4  # Number of running jobs
    # partition = 'gpu'
    jobs = [str(i) for i in range(njobs)]
    sbatch_f = """#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=jobs/log/array_%A_%a.log
#SBATCH --array=0-{njobs}%{ntasks}
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH -p {partition}
#SBATCH --ntasks=1

date; hostname; pwd
python -c "import torch; print(torch.__version__)"
(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step
source $HOME/export_p1.sh
bash jobs/$SLURM_ARRAY_TASK_ID.sh
""".format(njobs=njobs-1, ntasks=ntasks, partition=partition)
    with open('jobs/slurm.sbatch', 'w') as f:
        print(sbatch_f, file=f)
    parallel = True  # each script runs in parallel
    print('Total jobs: %d, Array jobs: %d, Max active: %d, Partition: %s'
          % (num_runs, njobs, min(njobs, ntasks), partition))
    return jobs, parallel

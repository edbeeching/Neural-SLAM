#!/bin/bash
#!/bin/bash
#SBATCH --job-name=train_anm           # name of job
#SBATCH -A ugk@gpu
#SBATCH --partition=gpu_p1           # GPU partition requested
#SBATCH --ntasks-per-node 4
#SBATCH --nodes=1                    # number of nodes
#SBATCH --ntasks=4                    # number of processes (only one process here)
#SBATCH --gres=gpu:4                  # number of GPUs to reserve (only one GPU here)
#SBATCH --cpus-per-task=10            # number of cores to reserve (a quarter of the node)

#SBATCH --mem=160G		           # memory
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread          # reserves physical cores (and not logical cores)
#SBATCH --time=20:00:00               # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logs/train%j.out     # name of output file
#SBATCH --error=logs/train%j.err      # name of error file (here, in common with the output file)
 
# cleans out the modules loaded in interactive and inherited by default 
module purge
# loading of modules
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
module load gcc/9.1.0
source /gpfswork/rech/rwh/utv52ia/work/venvs/nslam/bin/activate

cd /gpfsdswork/projects/rech/rwh/utv52ia/work/Neural-SLAM

python test_env.py


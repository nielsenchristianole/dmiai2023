#!/bin/sh 


### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err

### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=1"

### -- set the job Name -- 
#BSUB -J tumor_segment

### -- ask for 1 core -- 
#BSUB -n 8

### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=3GB]"
#BSUB -R "span[hosts=1]"

### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB

### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 

# load CUDA (for GPU support)
module load cuda/11.3
module load python3/3.9.11

python3 -m venv venv_1

source venv_1/bin/activate

python3 -m pip install -r requirements.txt

python train.py
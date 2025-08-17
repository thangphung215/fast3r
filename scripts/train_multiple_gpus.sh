set -e 
# pip install -r requirements.txt
# pip install efficientnet-pytorch timm transformers
# 5090 pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129

export SLURM_NODEID=0
export SLURM_NTASKS=2
export SLURM_NTASKS_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355


export SLURM_LOCALID=0
export SLURM_PROCID=0
export LOCAL_RANK=0

# Debug info
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


HYDRA_FULL_ERROR=1 python fast3r/train.py experiment=super_long_training/super_long_training &


export SLURM_LOCALID=1
export SLURM_PROCID=1
export LOCAL_RANK=1

# Debug info
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


HYDRA_FULL_ERROR=1 python fast3r/train.py experiment=super_long_training/super_long_training

set -e 
# pip install -r requirements.txt
# pip install efficientnet-pytorch timm transformers
# 5090 pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
export SLURM_NODEID=0
export SLURM_LOCALID=0  # Usually 0 for single-process local runs
export SLURM_NODEID=0   # Usually 0 for single-node local runs
export SLURM_PROCID=0
export SLURM_NTASKS=1

HYDRA_FULL_ERROR=1 python fast3r/train.py experiment=super_long_training/super_long_training
# python fast3r/resume_train.py logs/super_long_training_mobilenetv4_arkit_2inputs/runs/super_long_training_mobilenetv4_arkit_2inputs_99999_20250716
# HYDRA_FULL_ERROR=1 python fast3r/train.py experiment=demo_training/demo_training
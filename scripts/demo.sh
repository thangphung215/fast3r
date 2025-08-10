export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$PATH:$(pwd)


checkpoint_dir_raw=logs/super_long_training_mobilenetv4_arkit_2inputs_test2_sum/runs/super_long_training_mobilenetv4_arkit_2inputs_test2_sum_99999_20250716
checkpoint_dir=lightfast3rv2_Checkpoint

# python fast3r/utils/checkpoint_utils.py
python fast3r/viz/demo.py --checkpoint_dir $checkpoint_dir 
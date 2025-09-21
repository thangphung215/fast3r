mkdir data

ROOT_VAL=/workspace/gdfast/val
ROOT_TRAIN=/workspace/gdfast/train
ROOT_CKPT=/workspace/gdfast/ckpts

ln -s $ROOT_VAL/7scenes data/7_scenes_processed
ln -s $ROOT_VAL/NeuralRGBD data/neural_rgbd
# ln -s $ROOT_VAL/DTU/dtu_test_mvsnet_release/ data/dtu_test_mvsnet_release

ln -s $ROOT_TRAIN/arkitscenes_processed data/arkitscenes_processed
# ln -s $ROOT_TRAIN/preprocessed data/co3d_all_seqs_per_category_subset_processed
ln -s $ROOT_TRAIN/scannetpp_processed data/scannetpp_processed

ln -s $ROOT_CKPT ckpts
# conda env create -f fast3r_env.yml
export PYTHONPATH=$(pwd)/fast3r:$PYTHONPATH

RAW_DIR=data/scannetpp_raw
PAIRS=/hdd6/ScanNetpp/scannetpp_pairs
python3 datasets_preprocess/preprocess_scannetpp_2.py \
    --scannetpp_dir $RAW_DIR \
    --precomputed_pairs $PAIRS \
    --pyopengl-platform egl

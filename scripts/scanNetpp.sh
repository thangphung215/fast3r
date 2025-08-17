export PYTHONPATH=$(pwd)/fast3r:$PYTHONPATH

RAW_DIR=/media/thangphung/hdd7/ScanNetPP/
PAIRS=/hdd6/ScanNetpp/scannetpp_v2_pairs/
# python3 datasets_preprocess/preprocess_scannetpp.py \
python3 datasets_preprocess/preprocess_scannetpp_2.py \
    --scannetpp_dir $RAW_DIR \
    --precomputed_pairs $PAIRS \
    --pyopengl-platform egl

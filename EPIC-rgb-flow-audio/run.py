import os

cmd1="python train_video_flow_audio_EPIC_MOOSA.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --mask_ratio 0.7 --entropy_min_weight 0.001 --jigsaw_num_splits 2 --datapath D:\ML\Dataset\EPIC_KITCHENS --train_round _r1"

os.system(cmd1)
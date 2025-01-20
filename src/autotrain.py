import os
os.system(('CUDA_VISIBLE_DEVICES=1 python train.py --model ALECE --data STATS --batch_size 64 --wl_type upd_heavy --use_query_bitmap 1 --attn_head_key_dim 2048 --bitmap_size 100 --n_epoch 80 --experiments_dir ../res/20250118/use_query_bitmap_100_epoch80_attn_head_key_dim_2048'))

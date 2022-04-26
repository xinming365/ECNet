#!/usr/bin/env bash
 python forces_trainer_heanet.py --lr 1e-3 --epochs 1 --batch_size 128 --weight_decay 1e-4 --hidden_channels 128 --n_filters 64 --n_interactions 3   --cutoff 8 -t  --is_validate
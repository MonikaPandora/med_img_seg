#! /bin/sh
cd ..
python train.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 10 \
--epochs 100 \
--lr 0.0001 \
--lr-update 'CosineAnnealingWarmRestarts' \
--decoder_attention \
--folds 5
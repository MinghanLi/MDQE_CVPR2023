CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
       --config-file configs/R50_ovis_360.yaml \
       INPUT.PRETRAIN_FRAME_NUM 4 \
       INPUT.SAMPLING_FRAME_NUM 4 \

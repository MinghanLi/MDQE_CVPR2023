CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
      --config-file configs/R50_ovis_360.yaml \
	    --num-gpus 8 \
      --eval-only \
      MODEL.WEIGHTS output/mdqe_r50_ovis_bs16_360p_f4.pth \
	    INPUT.SAMPLING_FRAME_NUM 4 \
      OUTPUT_DIR output/ovis/mdqe_r50_ovis_bs16_360p_f4/ \

export CUDA_VISIBLE_DEVICES=0
python3 main_supervised.py \
--dataroot_A './fastMRI/full' \
--dataroot_B './fastMRI/down4_1D' \
--nEpoch 300 \
--decay_epoch 150 \
--lr 2e-4 \
--disp_div_N 30 \
--batch_size 1 \
--nC 1 \
--nX 640 \
--ngf 32 \
--norm 'instance' \
--beta1 0.5 \
--beta2 0.99 \
--is_training \
--generator 'unet' \
--gpu_ids '0' \
--savepath './Results/cycleGAN_AccMRI' \
--name '1D_DS4_unet_supervised'

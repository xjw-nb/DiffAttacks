#!/usr/bin/env bash
cd ../..

SEED1=0
SEED2=1234

for t in 75; do
  for adv_eps in 0.5; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=0 python eval_sde_adv.py --exp ./exp_results --config cifar10.yml \
          -i MSE_L2_70_cifar10-robust_adv-$t-eps$adv_eps-64x1-bm0-t0-end1e-5-cont-L2-eot20 \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 128 \
          --num_sub 128 \
          --domain cifar10 \
          --classifier_name cifar10-wrn-70-16-dropout \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type ddpm_cifar \
          --attack_version rand \
          --eot_iter 20 \
          --lp_norm L2 \

      done
    done
  done
done

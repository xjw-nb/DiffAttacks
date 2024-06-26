#!/usr/bin/env bash
cd ../..

SEED1=0
SEED2=1234

for t in 100; do
  for adv_eps in 0.031373; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=0 python eval_sde_adv.py --exp ./exp_results --config cifar10.yml \
          -i diffatack_cifar10-robust_adv-$t-eps$adv_eps-64x1-bm0-t0-end1e-5-cont-wres70-16-eot20 \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 1 \
          --num_sub 64 \
          #--domain imagenet \
         --domain cifar10 \
         --classifier_name cifar10-wrn-70-16-dropout \
         #--classifier_name imagenet-resnet50 \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --score_type score_sde \
          --attack_version rand \
          --eot_iter 20

      done
    done
  done
done

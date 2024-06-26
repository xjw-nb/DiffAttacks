#!/usr/bin/env bash
cd ../..

SEED1=1234
SEED2=0

for t in 150; do
  for adv_eps in 0.0157; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=0 python eval_sde_adv.py --exp ./exp_results --config imagenet.yml \
          -i diffattack_adv-$t-eps$adv_eps-4x4-bm0-t0-end1e-5-cont-eot20 \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 1 \
          --num_sub 1 \
          --domain imagenet \
          --classifier_name imagenet-deit-s \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --attack_version rand \
          --eot_iter 20

      done
    done
  done
done


#!/user/bi
NUM_PROC=8
torchrun --nproc_per_node=$NUM_PROC roundwise_trainer.py \
  --model deit_small_patch16_224 \
  --pretrained \
  --epochs 300 \
  --opt adamW \
  --sched inverse_sqrt \
  --warmup-epochs 5 \
  --warmup-lr 1e-6 \
  --warmup_init_lr 1e-6\
  --warmup_updates 4000 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --aa rand-m9-mstd0.5 \
  --smoothing 0.2 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --model-ema \
  --model-ema-decay 0.9999 \
  --num-classes 1000 \
  --output ./ \
  --workers 1 \
  --recovery-interval 1251 \
  --pretrain_ck checkpoint_10.pth  \
  --start_epoch 10 \
  --moe_epoch 30 \
  --num_round 1 \
  --finetinue_epoch 10 \
  --batch-size 128 \
  --data-path /ssd1/xinyu/datasets/imagenet_object_localization_challenge/ILSVRC/Data/CLS-LOC/ \
  --output_dir ./10_30_10/


#python -m torch.distributed.launch --nproc_per_node=1 --use_env roundwise_trainer.py --model deit_small_patch16_224 --pretrain_ck checkpoint_10.pth  --start_epoch 10 --moe_epoch 30 --num_round 1 --finetinue_epoch 10 --batch-size 256 --data-path /scratch-shared/boqian/imagenet --output_dir ./
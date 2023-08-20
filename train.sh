
#!/user/bin

python -m torch.distributed.launch --nproc_per_node=1 --use_env roundwise_trainer.py --model deit_small_patch16_224 --pretrain_ck checkpoint_10.pth  --start_epoch 10 --moe_epoch 30 --num_round 1 --finetinue_epoch 10 --batch-size 256 --data-path /scratch-shared/boqian/imagenet --output_dir ./
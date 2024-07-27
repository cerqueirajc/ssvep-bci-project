#! /bin/sh

# nohup python -u scripts_v3/exp_benchmark_dataset_delayed.py > /mnt/mystorage/exp_benchmark_dataset_delayed.log &
# nohup python -u scripts_v3/exp_beta_dataset_delayed_01_15.py > /mnt/mystorage/exp_beta_dataset_delayed_01_15.log &
# nohup python -u scripts_v3/exp_beta_dataset_delayed_16_70.py > /mnt/mystorage/exp_beta_dataset_delayed_16_70.log &

nohup python -u scripts_v3/exp_benchmark_dataset_delayed_mp.py > /mnt/mystorage/exp_benchmark_dataset_delayed_mp.log &
nohup python -u scripts_v3/exp_beta_dataset_delayed_01_15_mp.py > /mnt/mystorage/exp_beta_dataset_delayed_01_15_mp.log &
nohup python -u scripts_v3/exp_beta_dataset_delayed_16_70_mp.py > /mnt/mystorage/exp_beta_dataset_delayed_16_70_mp.log &

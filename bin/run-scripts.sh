#! /bin/sh

nohup python scripts_v3/exp_benchmark_dataset_delayed.py -u > exp_benchmark_dataset_delayed.log &
nohup python scripts_v3/exp_beta_dataset_delayed_01_15.py -u > exp_beta_dataset_delayed_01_15.log &
nohup python scripts_v3/exp_beta_dataset_delayed_16_70.py -u > exp_beta_dataset_delayed_16_70.log &

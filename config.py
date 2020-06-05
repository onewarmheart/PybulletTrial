# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:10:22 2020

@author: Hill
"""
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # mode = 'train' or 'test'
    parser.add_argument('--mode', default='', type=str) 
    parser.add_argument('--dir_name', default='', type=str) 
    parser.add_argument("--env_name", default="Pendulum-v0")
    parser.add_argument("--unwrapped", default=True, type = bool)
    parser.add_argument("--HER", default=False, type = bool )
    parser.add_argument("--HER_sample_num", default=2, type = int )
    parser.add_argument("--pretrain", default=True, type = bool )
    parser.add_argument('--pretrain_episodes', default=120, type=int)
    parser.add_argument('--tau',  default=0.001, type=float) # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--evaluate_interval', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--buffer_size', default=1000000, type=int)
    parser.add_argument('--max_episodes', default=300, type=int)
    parser.add_argument('--max_steps', default=300, type=int)
    parser.add_argument('--test_steps', default=600, type=int) #  num of  games
    parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
    parser.add_argument('--seed', default=True, type=bool)
    parser.add_argument('--random_seed', default=1, type=int)
    parser.add_argument('--load_path', default='202003162217_mprlc_seed42', type=str)
    # optional parameters
    parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--log_interval', default=50, type=int) #
#    parser.add_argument('--load', default=False, type=bool) # load model
    parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
    parser.add_argument('--exploration_noise', default="ou", type=str)
    parser.add_argument('--update_iteration', default=10, type=int)
    parser.add_argument('--visdom', default=True, type=bool)
    parser.add_argument('--sigma', default=64.3, type=float) # std of gaussian noise
    args = parser.parse_args()

    return args
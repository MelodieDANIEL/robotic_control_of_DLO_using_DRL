"""
Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Mélodie HANI DANIEL ZAKARIA
PhD Student
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.hani_daniel_zakaria@uca.fr
"""

import sys
import gym
import numpy as np
import random
import torch
import os

import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

import argparse
import os
import time
from datetime import datetime

from mpi4py import MPI
import pybullet as p

import gym_panda_frite
from database_frite import Database_Frite

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test' or 'debug_cartesian' or 'debug_articular'
parser.add_argument("--env_name", default="PandaFrite-v1") # name of pybullet env
parser.add_argument('--log_interval', default=50, type=int) # frequency of saving neural weights, every 'log_interval' episodes
parser.add_argument('--max_episode', default=2000, type=int) # num of episodes
parser.add_argument('--max_step', default=500, type=int) # num of step per episodes
parser.add_argument('--batch_size', default=128, type=int) # size of the trained tuples
parser.add_argument('--max_memory_size', default=50000, type=int) # size of DDPG Agent memory
parser.add_argument('--random_seed', default=9527, type=int) # value to initialize the random number generator
parser.add_argument('--save_dir_name', default='./weights/', type=str) # name of neural weights directory (create it when it doesn't exist)
parser.add_argument('--cuda', default=False, type=bool) # use cuda
parser.add_argument('--generate_database_name', default='database_id_frite.txt', type=str) # name of generated 'goal' database (by default 'database_id_frite.txt')
parser.add_argument('--load_database_name', default='database_id_frite.txt', type=str) # name of loaded 'goal' database (by default 'database_id_frite.txt')
parser.add_argument('--distance_threshold', default=0.05, type=float) # distance necessary to success the goal
parser.add_argument('--generate_db_dir_name', default='/default_generate/', type=str) # directory name of generated database 
parser.add_argument('--load_db_dir_name', default='/default_load/', type=str) # directory name of 'goal' database loaded
parser.add_argument('--db_nb_x', default=8, type=int) # how to divide the 'goal space' on x to generate a 'goal' database
parser.add_argument('--db_nb_y', default=22, type=int) # how to divide the 'goal space' on y to generate a 'goal' database
parser.add_argument('--db_nb_z', default=10, type=int) # how to divide the 'goal space' on z to generate a 'goal' database
parser.add_argument('--gui', default=False, type=bool) # boolean to show or not the gui
parser.add_argument('--txt_values_path', default='./txt_values/', type=str)

args = parser.parse_args()

def main():

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    
    directory = args.save_dir_name
    txt_values_path = args.txt_values_path
    root_path_databases = "./databases"
    generate_path_databases = root_path_databases + args.generate_db_dir_name
    load_path_databases = root_path_databases + args.load_db_dir_name
    
    rank = MPI.COMM_WORLD.Get_rank()
    
    if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.isdir(txt_values_path):
               os.makedirs(txt_values_path)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            if not os.path.isdir(generate_path_databases):
                os.makedirs(generate_path_databases)
            if not os.path.isdir(load_path_databases):
                os.makedirs(load_path_databases)
    
    if not os.path.isfile(load_path_databases + args.load_database_name):
           raise RuntimeError("=> Database file to load does not exit : " + load_path_databases + args.load_database_name)
           return        
	
    db = Database_Frite(path_load=load_path_databases, load_name=args.load_database_name, generate_name=args.generate_database_name, path_generate=generate_path_databases, nb_x=args.db_nb_x, nb_y=args.db_nb_y, nb_z=args.db_nb_z)
    env = gym.make(args.env_name, database=db, distance_threshold=args.distance_threshold, gui=args.gui)
    
    env.seed(args.random_seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.random_seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.random_seed + MPI.COMM_WORLD.Get_rank())
    
    if (args.cuda):
        torch.cuda.manual_seed(args.random_seed + MPI.COMM_WORLD.Get_rank())
    
    agent = DDPGagent(args.cuda, env, max_memory_size=args.max_memory_size, directory=directory)
    noise = OUNoise(env.action_space)
    
    list_global_rewards = []
    
    if args.mode == 'test':
        print("mode test !")
        agent.load()
        n_episodes = 100
        n_steps = 10
        sum_distance_error = 0
        for episode in range(n_episodes):
            print("Episode : {}".format(episode))
           
            state = env.reset()
            if (args.gui):
               env.draw_env_box()
               time.sleep(0.5)
            current_distance_error = 0
            for step in range(n_steps):
                action = agent.get_action(state)
                print("action={}".format(action))
               
                new_state, reward, done, info = env.step(action)
                current_distance_error = info['mean_distance_error']
                if (args.gui):
                    env.draw_id_to_follow()
                
                print("step={}, distance_error={}".format(step,info['mean_distance_error']))
                #print("step={}, action={}, reward={}, done={}, info={}".format(step,action,reward, done, info))
                state = new_state
               
                if done:
                   print("done with step={}  !".format(step))
                   break
            if (args.gui):
                time.sleep(0.75)
            
           
            sum_distance_error += current_distance_error
        print("mean distance error = {}".format(sum_distance_error/n_episodes))
        print("sum distance error = {}".format(sum_distance_error))
		
    elif args.mode == 'train':
        start=datetime.now()
        
        if rank==0:
           f_mean_rewards = open(txt_values_path + "mean_rewards.txt", "w+")
           f_max_rewards = open(txt_values_path + "max_rewards.txt", "w+")
           f_min_rewards = open(txt_values_path + "min_rewards.txt", "w+")
           
        #print("begin mode train !")
        total_step = 0
        global_step_number = 0
        update_every_n_steps = 20
        for episode in range(args.max_episode):
            #print("** rank {}, episode {}".format(rank,episode))
            state = env.reset()
            noise.reset()
            episode_reward = 0
            for step in range(args.max_step):
                   action = agent.get_action(state)
                   action = noise.get_action(action, step)
                   new_state, reward, done, info = env.step(action) 
                   agent.memory.push(state, action, reward, new_state, done)
                   global_step_number += 1
                
                   if len(agent.memory) > args.batch_size:
                        agent.update(args.batch_size)
		
                   state = new_state
                   episode_reward += reward
                   
            #print('[{}] rank is: {}, episode is: {}, episode reward is: {:.3f}'.format(datetime.now(), rank, episode, episode_reward))
            
            
            global_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.SUM)/MPI.COMM_WORLD.Get_size()
            list_global_rewards.append(global_reward)
            
            min_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.MIN)
            max_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.MAX)
                 
            if rank == 0:
               f_mean_rewards.write("{} {:.3f}\n".format(episode,global_reward))
               f_min_rewards.write("{} {:.3f}\n".format(episode, min_reward ))
               f_max_rewards.write("{} {:.3f}\n".format(episode, max_reward))
	
               print('=> [{}] episode is: {}, eval success rate is: {:.3f}'.format(datetime.now(), episode, list_global_rewards[episode])) 
               if episode % args.log_interval == 0:
                  agent.save()
                  
        if rank == 0:          
           agent.save()
           f_mean_rewards.close()
           f_min_rewards.close()
           f_max_rewards.close()
           print("end mode train !")
           print("time elapsed = {}".format(datetime.now()-start))
        
        
    elif args.mode == 'debug_cartesian':	
        state = env.reset(use_frite=True)
        env.draw_env_box()
        env.show_cartesian_sliders()
        env.draw_all_ids_mesh_frite()
        while True:
            keys = p.getKeyboardEvents()
            
            env.apply_cartesian_sliders()
            env.draw_gripper_position()
            env.draw_id_to_follow()
            env.draw_text_gripper_position()
            env.draw_text_joints_values()
            
            
            
            if 65309 in keys:
               break 
    elif args.mode == 'debug_articular':	
        state = env.reset(use_frite=True)
        env.draw_env_box()
        env.show_sliders()
        while True:
            keys = p.getKeyboardEvents()
            
            env.apply_sliders()
            env.draw_gripper_position()
            env.draw_id_to_follow()
            env.draw_text_gripper_position()
            env.draw_text_joints_values()
           
            if 65309 in keys:
               break 
    elif args.mode == 'simple_test':
        state = env.reset(use_frite=True)
        env.draw_env_box()
        env.draw_gripper_position()
        print("first state = {}".format(state))
        env.draw_all_ids_mesh_frite()
        while True:
            keys = p.getKeyboardEvents()
            if 65309 in keys:
               break	       
    elif args.mode == 'generate_database':
        state = env.reset(use_frite=True)
        env.draw_env_box()
        
        db.print_config()
        
        time.sleep(2)
        
        db.generate()
        
        print("End !")
        
        while True:
            keys = p.getKeyboardEvents()
            if 65309 in keys:
               break
               
    elif args.mode == 'show_database':
        state = env.reset(use_frite=True)
        env.draw_env_box()
        
        db.print_config()
        db.debug_all_points()
        #db.debug_all_random_points(100)
        #print("random targets = {}".format(db.get_random_targets()))
        
        while True:
            keys = p.getKeyboardEvents()
            if 65309 in keys:
               break
               
    else:
        raise NameError("mode wrong!!!")
        
        
if __name__ == '__main__':
    main()


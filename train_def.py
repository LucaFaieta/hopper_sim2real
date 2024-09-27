import os
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper, VecEnv
from stable_baselines3.common.callbacks import BaseCallback 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
import random
import zipfile

# Add this import for making the env
from env.custom_hopper import *


OPT1 = True
OPT2 = True


MY_ADDS = True


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _init

def check_reward(reward, param_dict):
  if len(param_dict) == 0:
    return True
  upper_bound = None
  if len(param_dict)==1:
    return True

  sorted_dict = dict(sorted(param_dict.items(), key=lambda item: item[1][0]))

  dict_items_list = list(sorted_dict.items())

  middle_index = len(dict_items_list) // 2

  middle_key, middle_value = dict_items_list[middle_index]
  print(f"middle_vlaue is {middle_value}, reward is {reward}")

  if reward < middle_value[0]:
    return False
  return True

def parameters_from_numpy(env_source):

  list1 = env_source.env_method("get_M",indices = 0)[0]
  list2 = env_source.env_method("get_S",indices = 0)[0]

  tuple1 = tuple(list1)
  tuple2 = tuple(list2)

  hashable_tuple = (tuple1, tuple2)
  return hashable_tuple




def main():

    print('State space source enviroment:', env_source.observation_space)  # state-space
    print('Action space source enviroment:', env_source.action_space)  # action-space
    print('Dynamics parameters source enviroment:', env_source.get_parameters())  # masses of each link of the Hopper

    print('State space target enviroment:', env_target.observation_space)  # state-space
    print('Action space target enviroment:', env_target.action_space)  # action-space
    print('Dynamics parameters target enviroment:', env_target.get_parameters())


    log_dir_source = "./tmp/gym_source"
    log_dir_target = "./tmp/gym_target"
    os.makedirs(log_dir_source, exist_ok=True)
    os.makedirs(log_dir_target, exist_ok=True)

    num_workers= 8
    env_source = SubprocVecEnv([make_env('CustomHopper-source-autoinclined-v0', random.choice(range(120))) for i in range(num_workers)])
    env_target = SubprocVecEnv([make_env('CustomHopper-target-v0', i) for i in range(num_workers)])
    

    # Load the model
    #model_target = SAC.load("/content/drive/MyDrive/RL_hopper/SAC_target_1000000.zip")


    model_source = SAC("MlpPolicy", env_source, learning_rate=0.001, device="cuda")
    model_target = SAC("MlpPolicy", env_target, learning_rate=0.001, device="cuda" )


    rewards_log = []
    if OPT1:
      best_mean = 0      
      param_dict = {}
      for i in range(100):

          model_source.learn(total_timesteps=10000, progress_bar=True)
          mean_reward_st, std_reward_st = evaluate_policy(model_source, env_target, return_episode_rewards=True, n_eval_episodes=50)
          print(f"Test reward Source -> Target (avg +/- std): ({sum(mean_reward_st) / float(len(mean_reward_st))} +/- {sum(std_reward_st) / float(len(std_reward_st))}) - Num episodes: {50}")        
          avg_reward = sum(mean_reward_st) / float(len(mean_reward_st))
          avg_std = sum(std_reward_st) / float(len(std_reward_st))

          if OPT2:
            while not check_reward(avg_reward, param_dict):
                print(f"reward {avg_reward} still not good enough")
                model_source.learn(total_timesteps=10000, progress_bar=True)
                mean_reward_st, std_reward_st = evaluate_policy(model_source, env_target, return_episode_rewards=True, n_eval_episodes=50)
                avg_reward = sum(mean_reward_st) / float(len(mean_reward_st))
            
            temp_tuple = parameters_from_numpy(env_source)
            param_dict[temp_tuple] = (avg_reward, avg_std)

          print(f"iteration {i*10000}")   
          rewards_log.append(avg_reward)
          if avg_reward > best_mean:
            for i in range(num_workers):
              env_source.env_method("move_distribution",indices = i)       

    else:
       model_source.learn(total_timesteps=1_000_000, progress_bar=True, callback=save_learning_curve_callback)



    with open("/content/drive/MyDrive/RL_hopper/logs/MWB", "w") as file:
      file.write(str(rewards_log))
    
    print("Testing...")
    mean_reward_soso, std_reward_soso = evaluate_policy(model_source,env_source,return_episode_rewards= True, n_eval_episodes = 50)
    print(f"Test reward Source -> Source (avg +/- std): ({sum(mean_reward_soso)/float(len(mean_reward_soso))} +/- {sum(std_reward_soso)/float(len(std_reward_soso))}) - Num episodes: {50}")
    mean_reward_st, std_reward_st = evaluate_policy(model_source,env_target,return_episode_rewards= True,n_eval_episodes = 50)
    print(f"Test reward Source -> Target (avg +/- std): ({sum(mean_reward_st)/float(len(mean_reward_st))} +/- {sum(std_reward_st)/float(len(std_reward_st))}) - Num episodes: {50}")
    mean_reward_tt, std_reward_tt = evaluate_policy(model_target,env_target,return_episode_rewards= True,n_eval_episodes = 50)
    print(f"Test reward Target -> Target (avg +/- std): ({sum(mean_reward_tt)/float(len(mean_reward_tt))} +/- {sum(std_reward_tt)/float(len(std_reward_tt))}) - Num episodes: {50}")

if __name__ == '__main__':
  main()
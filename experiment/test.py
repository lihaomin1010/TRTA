import random
import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from tqdm import tqdm
import matplotlib.pyplot as plt

from env.online.mw_basketball import MWBasketball

from buffer.old import HER

from algo.policy_based import ddpg

if __name__ == '__main__':

    env = gym.make('FetchPush', max_episode_steps=50, render_mode='human')

    step = 0
    while True:
        done = False
        truncated = False
        step += 1
        success = 0
        env.reset()
        while not (done or truncated):
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            if reward > 0.0:
                success += 1
                f = open("/home/lihaomin/workspace/TRTA-new/success.txt", "a+")
                f.write("hello")
                f.write(str(success))
                f.write("\n")
                f.write(str(step))
                f.write("\n")
                f.close()
            if np.sum((state['achieved_goal']- state['desired_goal'])**2) < 0.005:
                success += 1
                f = open("/home/lihaomin/workspace/TRTA-new/success.txt", "a+")
                f.write("hello")
                f.write(str(success))
                f.write("\n")
                f.write(str(step))
                f.write("\n")
                f.close()
        if step % 100 == 0:
            print(f'look {step}')


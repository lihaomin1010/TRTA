import metaworld
import random

import numpy as np


class MWBasketball:
    def __init__(self, render=None):
        self.ml1 = metaworld.ML1('basketball-v2')  # Construct the benchmark, sampling tasks

        self.env = self.ml1.train_classes['basketball-v2'](render_mode=render)
        self.task = random.choice(self.ml1.train_tasks)
        self.env.set_task(self.task)  # Set task
        self.success = 0

    def reset(self):
        self.env.reset()  # Reset environment
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        return np.hstack([next_obs, info['target']])

    def step(self, action, render=False):
        # a = env.action_space.sample()  # Sample an action
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # info['target']
        # obs[4:7]

        if render:
            self.env.render()
        if reward >= 5.:
            self.success += 1
            print(self.success)
        return np.hstack([next_obs, info['target']]), reward, terminated or truncated


    def sample_action(self):
        return self.env.action_space.sample()


if __name__ == '__main__':
    env = MWBasketball()
    while True:
        env.reset()
        for step in range(10):
            a = env.sample_action()
            next_obs, reward, done = env.step(a)


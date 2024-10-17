import RND_HER_DDPG.rnd as rnd
import RND_HER_DDPG.ddpg as ddpg
import RND_HER_DDPG.buffer as buffer
import RND_HER_DDPG.agent as agent
import RND_HER_DDPG.utils as utils

import gymnasium as gym
import gymnasium_robotics

import numpy as np
import random

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":

    actor_lr = 1e-3
    critic_lr = 2e-3
    hidden_dim = 256
    action_bound = 1

    sigma = 0.1
    tau = 0.005
    gamma = 0.99

    num_episodes = 1000

    batch_size = 64 # must be x^2
    minimal_episodes = 50
    buffer_size = 100000

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ##############

    #env = gym.make('FetchReach-v3', render_mode='human')

    env = gym.make('FetchReach-v3')
    goal_dim = env.observation_space['achieved_goal'].shape[0]
    state_dim = env.observation_space['observation'].shape[0] + 2 * goal_dim
    action_dim = env.action_space.shape[0]

    intrinsic_policy = rnd.RNDModel(state_dim, device)

    extrinsic_policy = ddpg.DDPG(state_dim, hidden_dim, action_dim, action_bound,
                                 actor_lr, critic_lr, sigma, tau, gamma, device)

    replay_buffer = buffer.ReplayBuffer_Trajectory(capacity=10000, batch_size=batch_size)

    agent = agent.Agent(intrinsic_policy, extrinsic_policy, replay_buffer, env,  device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rr = []
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                traj = buffer.Trajectory(state)
                done = False
                truncated = False

                while not (done or truncated):
                    tr_state = buffer.trans_state(state)
                    action = agent.take_action(tr_state)
                    state, reward, done, truncated, _ = env.step(action)
                    rr.append(reward)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                #print(state['achieved_goal'],state['desired_goal'])
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)

                if replay_buffer.size() >= minimal_episodes:
                    agent.learn()
                # if replay_buffer.size() >= minimal_episodes:
                #    replay_buffer.sample(batch_size=64, use_her=True, task=env.env.env.env, batch=experience_buffer)
                #    for _ in range(64):
                #        current_experience_buffer = replay_buffer.small_sample(batch_size, experience_buffer)
                #        agent.update(current_experience_buffer)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return_mean = []
    for i in range(len(return_list)//20):
        return_mean.append(np.mean(return_list[20*i:20*i+20]))
    episodes_list = list(range(len(return_list)//20))
    plt.plot(episodes_list, return_mean)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG with HER on {}'.format('GridWorld'))
    plt.show()

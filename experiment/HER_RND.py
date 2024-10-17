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
    env = gym.make('FetchPush-v2')
    env.reset()

    actor_lr = 1e-3
    critic_lr = 2e-3
    hidden_dim = 256

    goal_dim = env.observation_space['achieved_goal'].shape[0]
    state_dim = env.observation_space['observation'].shape[0] + 2 * goal_dim
    action_dim = env.action_space.shape[0]
    action_bound = 1

    sigma = 0.1
    tau = 0.005
    gamma = 0.99
    num_episodes = 2000
    n_train = 20
    batch_size = 64
    ###############################
    minimal_rnd_episodes = 50
    minimal_her_episodes = 150
    buffer_size = 100000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    replay_buffer = HER.ReplayBuffer_Trajectory(buffer_size)
    experience_buffer = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
    agent = ddpg.DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                      critic_lr, sigma, tau, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rr = []
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                traj = HER.Trajectory(state)
                done = False
                truncated = False

                #max_steps = 500
                steps = 0

                while not (done or truncated):
                    steps += 1
                    o = state['observation']
                    ag = state['achieved_goal']
                    dg = state['desired_goal']
                    action = agent.take_action(np.concatenate((o, ag, dg)))
                    state, reward, done, truncated, _ = env.step(action)
                    rr.append(reward)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= minimal_episodes:
                    replay_buffer.sample(batch_size=64, use_her=True, task=env.env.env.env, batch=experience_buffer)
                    for _ in range(64):
                        current_experience_buffer = replay_buffer.small_sample(batch_size, experience_buffer)
                        agent.update(current_experience_buffer)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG with HER on {}'.format('GridWorld'))
    plt.show()

import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib. pyplot as plt

from env.online.mw_basketball import MWBasketball

from buffer.old import HER

from algo.policy_based import ddpg

if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-3
    hidden_dim = 256

    goal_dim = 3
    state_dim = 39 + goal_dim
    action_dim = 4
    action_bound = 1

    sigma = 0.1
    tau = 0.005
    gamma = 0.98
    num_episodes = 2000
    n_train = 20
    batch_size = 512
    ###############################
    minimal_episodes = 1
    buffer_size = 10000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = MWBasketball()
    env.reset()

    replay_buffer = HER.ReplayBuffer_Trajectory(buffer_size)
    agent = ddpg.DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                 critic_lr, sigma, tau, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rr = []
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = HER.Trajectory(state)
                done = False

                max_steps = 500
                steps = 0

                while not done and steps < max_steps:
                    steps += 1
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    rr.append(reward)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= minimal_episodes:
                    for _ in range(n_train):
                        transition_dict = replay_buffer.sample(batch_size, True)
                        agent.update(transition_dict)
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
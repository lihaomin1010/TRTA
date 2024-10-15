import random
import numpy as np
import collections

class Trajectory:
    ''' 用来记录一条完整轨迹 '''
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, task, batch, dis_threshold=0.08, her_ratio=0.8, k=4):
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]
            for _ in range(k):
                step_state = np.random.randint(traj.length)
                state = np.concatenate((traj.states[step_state]['observation'], traj.states[step_state]['achieved_goal'], traj.states[step_state]['desired_goal']))
                next_state = np.concatenate((traj.states[step_state+1]['observation'], traj.states[step_state+1]['achieved_goal'], traj.states[step_state+1]['desired_goal']))
                action = traj.actions[step_state]
                reward = traj.rewards[step_state]
                done = traj.dones[step_state]

                if use_her and np.random.uniform() <= her_ratio:
                    step_goal = np.random.randint(step_state, traj.length)
                    dg = traj.states[step_goal+1]['achieved_goal']


                    reward = task.compute_reward(dg, dg, 1.0)
                    done = True
                    state = np.concatenate((traj.states[step_goal]['observation'], traj.states[step_goal]['achieved_goal'], dg))
                    next_state = np.concatenate((traj.states[step_goal+1]['observation'], dg, dg))

                batch['states'].append(state)
                batch['next_states'].append(next_state)
                batch['actions'].append(action)
                batch['rewards'].append(reward)
                batch['dones'].append(done)

            #batch['states'] = np.array(batch['states'])
            #batch['next_states'] = np.array(batch['next_states'])
            #batch['actions'] = np.array(batch['actions'])
        return batch

    def small_sample(self, batch_size, buffer):
        idx = np.random.choice(len(buffer['dones']), batch_size, replace=False)
        new_buffer = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for i in idx:
            new_buffer['states'].append(buffer['states'][i])
            new_buffer['actions'].append(buffer['actions'][i])
            new_buffer['rewards'].append(buffer['rewards'][i])
            new_buffer['dones'].append(buffer['dones'][i])
            new_buffer['next_states'].append(buffer['next_states'][i])
        return new_buffer


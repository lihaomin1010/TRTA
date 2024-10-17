import math
import random
import numpy as np
import collections


class Trajectory:
    ''' 用来记录一条完整轨迹 '''

    def __init__(self, init_state):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.achieved_goals = []
        self.desired_goals = []
        self.length = 0

        s, ag, dg = split_state(init_state)
        self.states.append(s)
        self.achieved_goals.append(ag)
        self.desired_goals.append(dg)

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        s, ag, dg = split_state(state)
        self.states.append(s)
        self.achieved_goals.append(ag)
        self.desired_goals.append(dg)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1

    def size(self):
        return len(self.rewards)


class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''

    def __init__(self, capacity, batch_size=64):
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def simple_sample(self):
        k = int(math.sqrt(self.batch_size))
        idx = np.random.choice(len(self.buffer), k, replace=False)

        result = dict(states=[],
                      actions=[],
                      next_states=[],
                      rewards=[],
                      dones=[])
        for i in idx:
            traj = self.buffer[i]
            idx_s = np.random.choice(traj.size()-1, k, replace=False)
            for j in idx_s:
                result['states'].append(np.concatenate((traj.states[j], traj.achieved_goals[j], traj.desired_goals[j])))
                result['actions'].append(traj.actions[j])
                result['rewards'].append(traj.rewards[j])
                result['dones'].append(traj.dones[j])
                result['next_states'].append(np.concatenate((traj.states[j+1], traj.achieved_goals[j+1], traj.desired_goals[j+1])))
        return result

    def her_sample(self, env, her_ratio=0.8, k=4):
        batch = dict(states=[],
                      actions=[],
                      next_states=[],
                      rewards=[],
                      dones=[])
        for _ in range(self.batch_size):
            traj = random.sample(self.buffer, 1)[0]
            for _ in range(k):
                step_state = np.random.randint(traj.length)
                state = np.concatenate((traj.states[step_state], traj.achieved_goals[step_state], traj.desired_goals[step_state]))
                next_state = np.concatenate((traj.states[step_state+1], traj.achieved_goals[step_state+1], traj.desired_goals[step_state+1]))
                action = traj.actions[step_state]
                reward = traj.rewards[step_state]
                done = traj.dones[step_state]

                if np.random.uniform() <= her_ratio:
                    step_goal = np.random.randint(step_state, traj.length)
                    dg = traj.achieved_goals[step_goal + 1]

                    reward = env.env.env.env.compute_reward(dg, dg, 1.0)
                    done = True
                    action = traj.actions[step_goal]
                    state = np.concatenate((traj.states[step_goal], traj.achieved_goals[step_goal], dg))
                    next_state = np.concatenate((traj.states[step_goal + 1], dg, dg))

                batch['states'].append(state)
                batch['next_states'].append(next_state)
                batch['actions'].append(action)
                batch['rewards'].append(reward)
                batch['dones'].append(done)

        return batch


def split_state(state):
    o = state['observation']
    ag = state['achieved_goal']
    dg = state['desired_goal']
    return o, ag, dg


def trans_state(state):
    o = state['observation']
    ag = state['achieved_goal']
    dg = state['desired_goal']
    return np.concatenate((o, ag, dg))

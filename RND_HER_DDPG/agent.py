import torch
import numpy as np
import torch.nn.functional as F

class Agent:
    def __init__(self, in_policy, ex_policy, buffer, env, device, train_n=32):
        self.in_policy = in_policy
        self.ex_policy = ex_policy
        self.buffer = buffer
        self.env = env
        self.device = device
        self.epi = 0
        self.train_n = train_n
        self.use_her = False
        self.use_rnd = False
        self.use_weight_sample = True

    # for test data
    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.ex_policy.actor(state).detach().cpu().numpy()[0]
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.ex_policy.actor(state).detach().cpu().numpy()[0]
        # 给动作添加噪声，增加探索
        action = action + self.ex_policy.sigma * np.random.randn(self.ex_policy.action_dim)
        return action

    def learn(self):
        for _ in range(self.train_n):
            if self.use_her:
                trans_dict = self.buffer.her_sample()
            elif self.use_weight_sample:
                trans_dict = self.buffer.weight_sample()
            else:
                trans_dict = self.buffer.simple_sample()
            self.update(trans_dict)
    def update(self, transition_dict):
        self.epi += 1
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)

        if self.use_rnd:
            intrinsic_reward = self.in_policy.compute_intrinsic_reward(torch.stack((states, actions)), next_states)

            rewards += 10*intrinsic_reward

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.ex_policy.target_critic(next_states,
                                           self.ex_policy.target_actor(next_states))
        q_targets = rewards + self.ex_policy.gamma * next_q_values * (1 - dones)
        # MSE损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.ex_policy.critic(states, actions), q_targets))
        self.ex_policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.ex_policy.critic_optimizer.step()

        # 训练 RND
        if self.use_rnd:
            self.in_policy.update(torch.stack((states, actions)), next_states)

        # 策略网络就是为了使Q值最大化
        actor_loss = -torch.mean(self.ex_policy.critic(states, self.ex_policy.actor(states)))
        self.ex_policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.ex_policy.actor_optimizer.step()

        self.ex_policy.soft_update(self.ex_policy.actor, self.ex_policy.target_actor)  # 软更新策略网络
        self.ex_policy.soft_update(self.ex_policy.critic, self.ex_policy.target_critic)  # 软更新价值网络

        #if self.epi % 50 == 0:
        #    self.ex_policy.critic.initialize_weights()
            #self.ex_policy.actor.reset_parameters()
        #for ir in intrinsic_reward:
        #    if ir[0] > 0.001:
        #        print('Episode: {}, ir: {}'.format(self.epi, ir))
        #if self.epi % 100 == 1:
        #    print('Episode: {}, in-loss: {}'.format(self.epi, intrinsic_reward))
        return None
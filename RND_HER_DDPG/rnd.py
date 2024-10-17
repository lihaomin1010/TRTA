import torch.nn as nn
import torch
from torch.optim import Adam


class RND_Model(nn.Module):
    def __init__(self, state_dim, device):
        super(RND_Model, self).__init__()
        self.device = device
        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        ).float().to(device)

    def forward(self, states):
        return self.nn_layer(states)


class RNDModel:
    def __init__(self, state_dim, device, learning_rate=1e-3):
        self.RND_epochs = 5
        self.device = device
        self.rnd_predict = RND_Model(state_dim, device)
        self.rnd_predict_optimizer = Adam(self.rnd_predict.parameters(), lr=learning_rate)
        self.epi = 0

    def get_rnd_loss(self, state_pred, state_target):
        # Don't update target state value
        state_target = state_target.detach()

        # Mean Squared Error Calculation between state and predict
        forward_loss = torch.mean((state_target - state_pred).pow(2), dim=1).sum()
        return forward_loss

    def compute_intrinsic_reward(self, states, next_states):
        state_pred = self.rnd_predict(states)

        return torch.mean((next_states - state_pred).pow(2), dim=1).view(states.shape[0], 1)

    def update(self, states, next_states):
        for _ in range(self.RND_epochs):
            states_pred = self.rnd_predict(states)
            loss = self.get_rnd_loss(states_pred, next_states)

            self.rnd_predict_optimizer.zero_grad()
            loss.backward()
            self.rnd_predict_optimizer.step()
            # if loss.item() > 0.01:
            #    print(f'RND Loss: {loss.item()}, epi {self.epi}')
            self.epi += 1

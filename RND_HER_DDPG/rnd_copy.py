import torch.nn as nn
from torch.optim import Adam



class RND_Model(nn.Module):
    def __init__(self, state_dim, device):
        super(RND_Model, self).__init__()
        self.device = device
        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).float().to(device)

    def forward(self, states):
        return self.nn_layer(states)


class RNDModel:
    def __init__(self, state_dim, device, learning_rate=1e-3):
        self.RND_epochs = 5
        self.device = device
        self.rnd_predict = RND_Model(state_dim, device)
        self.rnd_predict_optimizer = Adam(self.rnd_predict.parameters(), lr=learning_rate)
        self.rnd_target = RND_Model(state_dim, device)

    def get_rnd_loss(self, state_pred, state_target):
        # Don't update target state value
        state_target = state_target.detach()

        # Mean Squared Error Calculation between state and predict
        forward_loss = ((state_target - state_pred).pow(2) * 0.5).mean()
        return forward_loss

    def compute_intrinsic_reward(self, states):
        state_pred = self.rnd_predict(states)
        state_target = self.rnd_target(states)

        return ((state_target - state_pred).pow(2)).mean()

    def update(self, states):
        for _ in range(self.RND_epochs):
            state_pred = self.rnd_predict(states)
            state_target = self.rnd_target(states)
            loss = self.get_rnd_loss(state_pred, state_target)

            self.rnd_predict_optimizer.zero_grad()
            loss.backward()
            self.rnd_predict_optimizer.step()

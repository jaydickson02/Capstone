import torch
import numpy as np
from models.q_network import QNetwork
from agents.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, hyperparams):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(self.state_dim, self.action_dim, hyperparams["hidden_dims"]).to(self.device)
        self.target_q_network = QNetwork(self.state_dim, self.action_dim, hyperparams["hidden_dims"]).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=hyperparams["learning_rate"])
        self.replay_buffer = ReplayBuffer(hyperparams["buffer_size"])

        self.gamma = hyperparams["gamma"]
        self.tau = hyperparams["tau"]
        self.epsilon = hyperparams["epsilon_start"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.min_epsilon = hyperparams["min_epsilon"]

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_q_network(next_state).max(1)[0]
        target_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network()

    def update_target_network(self):
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, checkpoint_path):
        torch.save(self.q_network.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.q_network.load_state_dict(torch.load(checkpoint_path))

    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon_decay * self.epsilon, self.min_epsilon)

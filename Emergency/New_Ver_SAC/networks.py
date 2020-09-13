import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions import Categorical
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='Critic', chkpt_dir='temp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # CUDA
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action] ,dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='Value', chkpt_dir='temp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # CUDA
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2,
                 name='Actor', chkpt_dir='temp/sac', control_strategy='Continuouse'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.control_strategy = control_strategy

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.max_action = max_action        # ???
        self.reparam_noise = 1E-6           # ???

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        if self.control_strategy == 'Continuouse':
            self.mu = nn.Linear(self.fc2_dims, self.n_actions)
            self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        else:
            self.act_prob = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # CUDA
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        if self.control_strategy == 'Continuouse':
            mu = self.mu(prob)
            sigma = self.sigma(prob)

            # 저자는 sigma 가 너무 작거나 커지는 것을 방지하기 위해 clamp 사용
            sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

            return mu, sigma
        else:
            prob = self.act_prob(state)
            action_probs = F.softmax(prob)
            return action_probs

    def sample_normal(self, state, reparameterize=True):
        if self.control_strategy == 'Continuouse':
            mu, sigma = self.forward(state)
            probabilities = Normal(mu, sigma)

            if reparameterize:
                actions = probabilities.rsample()
            else:
                actions = probabilities.sample()

            action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
            log_probs = probabilities.log_prob(actions)
            log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)

            log_probs = log_probs.sum(1, keepdim=True)  # Loss 구하기 위해서

            # 사용안함.
            action_probs, greedy_actions = 0, 0
        else:
            action_probs = self.forward(state)
            greedy_actions = T.argmax(action_probs, dim=1, keepdim=True)

            categorical = Categorical(action_probs)
            action = categorical.sample().view(-1, 1)

            log_probs = T.log(action_probs + (action_probs == 0.0).float() * 1e-8)

        return action, log_probs, action_probs, greedy_actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
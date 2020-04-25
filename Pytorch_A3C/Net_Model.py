import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent_network(nn.Module):
    def __init__(self, state_dim, state_time, action_dim):
        super(Agent_network, self).__init__()
        self.state_dim = state_dim
        self.state_time = state_time
        self.action_dim = action_dim

        self.test_data = torch.empty((state_dim, state_time))
        self.test_data = torch.empty((state_dim))

        # Build_net_structure
        # 1) build actor net
        # self.a_1 = torch.nn.Conv1d()
        self.a_1 = nn.Linear(state_dim, 2)

        print(self.forward(self.test_data))

    def forward(self, x):
        a = self.a_1(x)
        return a

if __name__ == '__main__':
    Agent_network(5, 4, 2)

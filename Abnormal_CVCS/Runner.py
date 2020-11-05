'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
https://github.com/quantumiracle/SOTA-RL-Algorithms
'''

import random

import gym
import numpy as np
import time

import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

import argparse

import torch.multiprocessing as mp

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from Abnormal_CVCS.CNS_CVCS import ENVCNS
from MONITORINGTOOL_CVCS import MonitoringMEM, Monitoring

GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.tot_ep = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def get_length(self):
        return len(self.buffer)

    def add_ep(self):
        self.tot_ep += 1

    def get_ep(self):
        return self.tot_ep


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
            action.detach().cpu().numpy()[0]
        # return action, mean.item(), std.item()
        return action, 0, 0

    def sample_action(self, state):
        # Ep 초기 랜덤 액션 용
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        _, mean, std = self.get_action(state, deterministic=True)
        return self.action_range * a.numpy(), mean, std


class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss)

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean(), q_value_loss1.tolist(), q_value_loss2.tolist(), policy_loss.tolist()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def worker(id, sac_trainer, ENV, rewards_queue, q1_queue, q2_queue, p_queue, Monitoring_ENV, \
           replay_buffer, max_episodes, max_steps, batch_size, explore_steps, \
           update_itr, AUTO_ENTROPY, DETERMINISTIC, hidden_dim, model_path):
    '''
    the function for sampling with multi-processing
    '''

    # print(sac_trainer, replay_buffer)
    # sac_tainer are not the same, but all networks and optimizers in it are the same; replay  buffer is the same one.
    if ENV == 'Pendulum':
        env = gym.make("Pendulum-v0")
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        action_range = 1.
    elif ENV == 'CNS':
        env = ENVCNS(Name=id, IP='192.168.0.103', PORT=int(f'710{id + 1}'), Monitoring_ENV=Monitoring_ENV)
        action_dim = env.action_space
        state_dim = env.observation_space
        action_range = 1.

    frame_idx = 0
    rewards = []
    # training loop
    for eps in range(max_episodes):
        replay_buffer.add_ep()
        episode_reward = 0
        episode_q1 = 0
        episode_q2 = 0
        episode_p = 0
        if ENV == 'Pendulum':
            state = env.reset()
        elif ENV == 'CNS':
            state = env.reset(file_name=f'{id}_{eps}')

        for step in range(max_steps):
            if frame_idx > explore_steps:
                action, mean_, std_ = sac_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC)
            else:
                action, mean_, std_ = sac_trainer.policy_net.sample_action(state)

            try:
                if ENV == 'Pendulum':
                    next_state, reward, done, _ = env.step(action)
                    # env.render()
                if ENV == 'CNS':
                    next_state, reward, done, action = env.step(action, mean_, std_)
            except KeyboardInterrupt:
                print('Finished')
                sac_trainer.save_model(model_path)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            # if len(replay_buffer) > batch_size:
            if replay_buffer.get_length() > batch_size:
                for i in range(update_itr):
                    _, episode_q1, episode_q2, episode_p = sac_trainer.update(batch_size,
                                                                              reward_scale=10.,
                                                                              auto_entropy=AUTO_ENTROPY,
                                                                              target_entropy=-1. * action_dim)
                    # print(episode_q1, episode_q2, episode_p)
                    # print(type(episode_q1), type(episode_q2), type(episode_p))
                    episode_q1 += episode_q1
                    episode_q2 += episode_q2
                    episode_p += episode_p

            # if eps % 10 == 0 and eps > 0:
            # plot(rewards, id)
            # sac_trainer.save_model(model_path)
            if done:
                print('Done')
                break
        print('Episode: ', replay_buffer.get_ep(), eps, '| Episode Reward: ', episode_reward, episode_q1, episode_q2,
              episode_p)
        # if len(rewards) == 0: rewards.append(episode_reward)
        # else: rewards.append(rewards[-1]*0.9+episode_reward*0.1)
        rewards_queue.put(episode_reward)
        q1_queue.put(episode_q1)
        q2_queue.put(episode_q2)
        p_queue.put(episode_p)

    sac_trainer.save_model(model_path)


def ShareParameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()


def plot(rewards, name):
    # clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig(f'sac_v2_multi{name}.png')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    num_workers = 3  # mp.cpu_count()    # TODO

    BaseManager.register('ReplayBuffer', ReplayBuffer)
    BaseManager.register('MonitoringMEM', MonitoringMEM)

    manager = BaseManager()
    manager.start()

    replay_buffer_size = 2e6
    replay_buffer = manager.ReplayBuffer(replay_buffer_size)  # share the replay buffer through manager
    #

    Monitoring_ENV = manager.MonitoringMEM(num_workers)

    # choose env
    ENV = ['Pendulum', 'CNS'][1]
    if ENV == 'Pendulum':
        env = gym.make("Pendulum-v0")
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        action_range = 1.
    elif ENV == 'CNS':
        env = ENVCNS(Name='GETINFO', IP='192.168.0.103', PORT=int(f'7100'))
        action_dim = env.action_space
        state_dim = env.observation_space
        action_range = 1.

    # hyper-parameters for RL training, no need for sharing across processes
    max_episodes = 10000
    max_steps = 4000 if ENV == 'CNS' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
    # max_steps = 12 if ENV == 'CNS' else 10  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
    batch_size = 128
    # batch_size = 10
    explore_steps = 0  # for random action sampling in the beginning of training
    update_itr = 1
    AUTO_ENTROPY = True
    DETERMINISTIC = False
    hidden_dim = 512
    model_path = './model'

    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range)

    # share the global parameters in multiprocessing
    sac_trainer.soft_q_net1.share_memory()
    sac_trainer.soft_q_net2.share_memory()
    sac_trainer.target_soft_q_net1.share_memory()
    sac_trainer.target_soft_q_net2.share_memory()
    sac_trainer.policy_net.share_memory()  # model
    sac_trainer.log_alpha.share_memory_()  # variable
    ShareParameters(sac_trainer.soft_q_optimizer1)
    ShareParameters(sac_trainer.soft_q_optimizer2)
    ShareParameters(sac_trainer.policy_optimizer)
    ShareParameters(sac_trainer.alpha_optimizer)

    rewards_queue = mp.Queue()  # used for get rewards from all processes and plot the curve
    q1_queue = mp.Queue()  # used for get rewards from all processes and plot the curve
    q2_queue = mp.Queue()  # used for get rewards from all processes and plot the curve
    p_queue = mp.Queue()  # used for get rewards from all processes and plot the curve

    processes = []
    rewards = []
    q1_q = []
    q2_q = []
    p_q = []


    for i in range(num_workers):
        process = Process(target=worker, args=(
            i, sac_trainer, ENV, rewards_queue, q1_queue, q2_queue, p_queue, Monitoring_ENV,
            replay_buffer, max_episodes, max_steps, batch_size, explore_steps,
            update_itr, AUTO_ENTROPY, DETERMINISTIC, hidden_dim, model_path))  # the args contain shared and not shared
        process.daemon = True  # all processes closed when the main stops

        processes.append(process)

    # MoProcess = Process(target=Monitoring, args=(Monitoring_ENV, ), daemon=True)
    # processes.append(MoProcess)

    if ENV == 'CNS':
        processes.append(Monitoring(Monitoring_ENV=Monitoring_ENV))
        pass

    [p.start() for p in processes]
    while True:  # keep geting the episode reward from the queue
        for Qu_, Qu_box in zip([rewards_queue, q1_queue, q2_queue, p_queue], [rewards, q1_q, q2_q, p_q]):
            GetedQu = Qu_.get()
            if GetedQu is not None:
                Qu_box.append(GetedQu)
            else:
                pass

        if len(rewards) % 20 == 0 and len(rewards) > 0:
            plot(rewards, name=f'{len(rewards)}R')
            plot(q1_q, name=f'{len(rewards)}q1')
            plot(q2_q, name=f'{len(rewards)}q2')
            plot(p_q, name=f'{len(rewards)}p_q')

    [p.join() for p in processes]  # finished at the same time

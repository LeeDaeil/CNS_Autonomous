"""
Builder: Daeil Lee 2021-01-03

Ref-Code:
    - https://github.com/ku2482/sac-discrete.pytorch
    -
"""
import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np
import asyncio

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime
from PZR_bubblegeneration_Fin.Memory import ReplayBuffer
from PZR_bubblegeneration_Fin.SAC_Network import ActorNet, CriticNet
from PZR_bubblegeneration_Fin.CNS_PZR import ENVCNS

from torch.utils.tensorboard import SummaryWriter
WRITER = SummaryWriter('./TFBoard')


class SAC:
    def __init__(self,
                 # info
                 net_type='DNN',
                 lr=0.0003, alpha=1, gamma=0.99, tau=0.005,

                 # mem_info
                 capacity=1e6, seq_len=2,

                 # Agent Run info
                 max_episodes=1e6, max_steps=1e6, interval_steps=15, target_update_interval=15,
                 batch_size=128,
                 ):
        # -----------------------------------------------------------------------------------------
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.interval_steps = interval_steps
        self.target_update_interval = target_update_interval
        # -----------------------------------------------------------------------------------------
        self._log_set()
        # -----------------------------------------------------------------------------------------
        # Call ENV
        self.envs, self.agent_n, self.a_dim, self.s_dim = self._call_env()

        # make Thread Pool
        self.pool = ThreadPoolExecutor(len(self.envs))

        #
        self.a_dim = 3

        # Define Memory
        self.replay_buffer = ReplayBuffer(capacity, net_type, seq_len)

        # Define Networks
        self.Actor_Policy_Nets = [ActorNet(nub_a=self.a_dim, nub_s=self.s_dim, net_type=net_type) for _ in self.envs]
        self.Critic_Q_Net1s = [CriticNet(nub_a=self.a_dim, nub_s=self.s_dim, net_type=net_type) for _ in self.envs]
        self.Critic_Q_Net2s = [CriticNet(nub_a=self.a_dim, nub_s=self.s_dim, net_type=net_type) for _ in self.envs]
        self.Critic_Q_Target_Net1s = [CriticNet(nub_a=self.a_dim, nub_s=self.s_dim, net_type=net_type) for _ in self.envs]
        self.Critic_Q_Target_Net2s = [CriticNet(nub_a=self.a_dim, nub_s=self.s_dim, net_type=net_type) for _ in self.envs]

        # Copy parameters from Critic_Q_Nets to Critoc_Q_Target_Nets
        for critic_q_target, critic_q, i in zip(self.Critic_Q_Target_Net1s, self.Critic_Q_Net1s, range(len(self.envs))):
            critic_q_target.load_state_dict(critic_q.state_dict())
            for critic_q_target_para in critic_q_target.parameters():
                critic_q_target_para.requires_grad = False
            critic_q.save(path=f'./Model/Critic_Q_net1_{i}')
        for critic_q_target, critic_q, i in zip(self.Critic_Q_Target_Net2s, self.Critic_Q_Net2s, range(len(self.envs))):
            critic_q_target.load_state_dict(critic_q.state_dict())
            for critic_q_target_para in critic_q_target.parameters():
                critic_q_target_para.requires_grad = False
            critic_q.save(path=f'./Model/Critic_Q_net2_{i}')

        # Save Models policy
        for poliy, i in zip(self.Actor_Policy_Nets, range(len(self.envs))):
            poliy.save(path=f'./Model/Actor_policy_net_{i}')

        # Define Optimizer
        self.Actor_Policy_Net_Opts = [opt.Adam(poliy_net.parameters(), lr=lr) for poliy_net in self.Actor_Policy_Nets]
        self.Critic_Q_Net1_Opts = [opt.Adam(critic_q_net1.parameters(), lr=lr) for critic_q_net1 in self.Critic_Q_Net1s]
        self.Critic_Q_Net2_Opts = [opt.Adam(critic_q_net2.parameters(), lr=lr) for critic_q_net2 in self.Critic_Q_Net2s]

        # Agent info ------------------------------------------------------------------------------
        print(f'{self.Actor_Policy_Nets}\n{self.Critic_Q_Net1s}\n{self.Critic_Q_Net2s}\n'
              f'{self.Critic_Q_Target_Net1s}\n{self.Critic_Q_Target_Net2s}')

        for i in range(self.agent_n):
            print(f'Agent {i}|'
                  f'ReplayBuffer {self.replay_buffer}|MonitoringMem {0}|'
                  f'ENV CNSIP{self.envs[i].CNS_ip}-CNSPort{self.envs[i].CNS_port}-'
                  f'ComIP{self.envs[i].Remote_ip}-ComPort{self.envs[i].Remote_port}')

        # Agent Run -------------------------------------------------------------------------------
        self._run(self.envs, self.replay_buffer, max_episodes, max_steps, interval_steps,
                  target_update_interval, batch_size)

    def _log_set(self):
        with open('Debug_logger.txt', 'w') as f: f.write(f'[{datetime.now()}]\n')

    def _log(self, txt):
        with open('Debug_logger.txt', 'a') as f: f.write(f'[{datetime.now()}]\t{txt}\n')

    def _call_env(self):
        _CNS_info = {
            0: ['192.168.0.211', 7101, False],           #CNS1
            1: ['192.168.0.211', 7102, False],
            2: ['192.168.0.211', 7103, False],
            3: ['192.168.0.211', 7104, False],
            4: ['192.168.0.211', 7105, False],
            #
            5: ['192.168.0.212', 7201, False],           #CNS2
            6: ['192.168.0.212', 7202, False],
            7: ['192.168.0.212', 7203, False],
            8: ['192.168.0.212', 7204, False],
            9: ['192.168.0.212', 7205, False],
            #
            10: ['192.168.0.213', 7301, False],           #CNS3
            11: ['192.168.0.213', 7302, False],
            12: ['192.168.0.213', 7303, False],
            13: ['192.168.0.213', 7304, False],
            14: ['192.168.0.213', 7305, False],
        }

        # Set CNS
        envs = [ENVCNS(Name=i, IP=_CNS_info[i][0], PORT=_CNS_info[i][1]) for i in range(len(_CNS_info))]
        return envs, len(_CNS_info), envs[0].action_space, envs[0].observation_space

    def _update(self, mini_batch, i, target_update):
        self._log(txt=f'call_update_{i}'+'='*50)
        s, a, r, s_next, d = mini_batch

        # print('_update_mini_batch:\n', s, s_next, a, r, d)
        s = torch.FloatTensor(s)
        s_next = torch.FloatTensor(s_next)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        d = torch.FloatTensor(np.float32(d)).unsqueeze(1)
        # print('_update:\n', s, s_next, a, r, d)

        # -------------------------------------------------------------------------------------
        # Update the Q-function or Critic network's parameters
        q1, q2 = self._update_cal_q(s, a, i)
        target_q = self._update_cal_target_q(r, s_next, d, i)

        Critic_Q1_loss = 0.5 * F.mse_loss(q1, target_q.detach())
        Critic_Q2_loss = 0.5 * F.mse_loss(q2, target_q.detach())
        self._log(txt=f'q1_{q1}_{target_q}')
        self._log(txt=f'q1_{q2}_{target_q}')

        Critic_Q1_loss_mean = torch.mean(Critic_Q1_loss)
        Critic_Q2_loss_mean = torch.mean(Critic_Q2_loss)

        # print(f'_Critic_loss_sum:\n{q1}\n{target_q}\n{Critic_Q1_loss}\n{Critic_Q2_loss}\n{Critic_Q1_loss_mean}')
        # print(f'_Critic_Q1_loss_mean:\n{Critic_Q1_loss_mean}')

        self.Critic_Q_Net1_Opts[i].zero_grad()
        Critic_Q1_loss_mean.backward()
        self.Critic_Q_Net1_Opts[i].step()

        self.Critic_Q_Net2_Opts[i].zero_grad()
        Critic_Q2_loss_mean.backward()
        self.Critic_Q_Net2_Opts[i].step()

        # -------------------------------------------------------------------------------------
        # Update the Actor_policy's parameters
        entropies, expect_q = self._update_cal_policy_entropy(s, i)

        Actor_policy_loss = entropies - expect_q
        Actor_policy_loss_mean = torch.mean(Actor_policy_loss)
        # print(f'Actor_policy_loss_mean:\n{Actor_policy_loss_mean}')

        self.Actor_Policy_Net_Opts[i].zero_grad()
        Actor_policy_loss_mean.backward()
        self.Actor_Policy_Net_Opts[i].step()

        # -------------------------------------------------------------------------------------
        # Log net calculation process
        ep = self.Wd[i]['ep']
        with open(f'./DB_ep_net/{ep}.txt', 'a') as f_net_:
            f_net_.write(f'{q1.data.tolist()}|{q2.data.tolist()}|{target_q.data.tolist()}|'
                         f'{Critic_Q1_loss_mean.data.tolist()}|{Critic_Q2_loss_mean.data.tolist()}|'
                         f'{entropies.data.tolist()}|{expect_q.data.tolist()}|'
                         f'{Actor_policy_loss.data.tolist()}|{Actor_policy_loss_mean.data.tolist()}|'
                         f'\n')

        # -------------------------------------------------------------------------------------
        # Update the Target Q network: soft-Q update
        if target_update:
            self._log(txt='target_update')
            Q_nets = [self.Critic_Q_Net1s[i], self.Critic_Q_Net2s[i]]
            Q_target_nets = [self.Critic_Q_Target_Net1s[i], self.Critic_Q_Target_Net2s[i]]

            for Q_net_, Q_target_net_ in zip(Q_nets, Q_target_nets):
                for Q_net_para_, Q_target_net_para_ in zip(Q_net_.parameters(), Q_target_net_.parameters()):
                    Q_target_net_para_.data.copy_(self.tau * Q_net_para_.data + (1 - self.tau) * Q_target_net_para_.data)

        return Critic_Q1_loss_mean.detach().cpu().numpy(), Critic_Q2_loss_mean.detach().cpu().numpy(), Actor_policy_loss_mean.detach().cpu().numpy()

    def _update_cal_q(self, s, a, i):
        q1 = self.Critic_Q_Net1s[i](s)
        q2 = self.Critic_Q_Net2s[i](s)
        q1 = q1.gather(1, a.long())
        q2 = q2.gather(1, a.long())
        return q1, q2

    def _update_cal_target_q(self, r, s_next, d, i):
        with torch.no_grad():
            Actor_s_next_out = self.Actor_Policy_Nets[i].sample(s_next)
            # print('_Actor_Policy_Net_next_Out:\n', Actor_s_next_out)
            action_next_, action_probs_next, log_probs_next = Actor_s_next_out

            q1_target = self.Critic_Q_Target_Net1s[i](s_next)
            q2_target = self.Critic_Q_Target_Net2s[i](s_next)

            min_q1_q2_target = torch.min(q1_target, q2_target)
            target_V = min_q1_q2_target - self.alpha * log_probs_next

            target_V_prob = action_probs_next * target_V
            target_V_sum = target_V_prob.sum(dim=1, keepdim=True)
            # print(f'_target_V:\n{min_q1_q2_target}\n{log_probs_next}\n{target_V}\n{action_probs_next}'
            #       f'\n{target_V_prob}\n{target_V_sum}')

        target_Q = r + self.gamma * (1 - d) * target_V_sum
        # print(f'_target_Q:\n{target_Q}')
        return target_Q

    def _update_cal_policy_entropy(self, s, i):
        Actor_s_out = self.Actor_Policy_Nets[i].sample(s)
        # print('_Actor_Policy_Net_Out:\n', Actor_s_out)
        action_, action_probs, log_probs = Actor_s_out

        with torch.no_grad():
            q1_ = self.Critic_Q_Net1s[i](s)
            q2_ = self.Critic_Q_Net2s[i](s)
            min_q1_q2 = torch.min(q1_, q2_)

        entropies = torch.sum(action_probs * self.alpha * log_probs, dim=1, keepdim=True)
        # print(f'_Actor_policy_entropies\n{action_probs}\n{log_probs}\n{entropies}')

        expect_q = torch.sum(action_probs * min_q1_q2, dim=1, keepdim=True)
        # print(f'_Actor_policy_expect_q\n{action_probs}\n{min_q1_q2}\n{expect_q}')

        return entropies, expect_q

    def _pool_one_step(self, envs, actions):
        def __pool_one_step(env, a):
            next_s, r, d, _ = env.step(a)
            return next_s, r, d, _

        futures = [self.pool.submit(__pool_one_step, env_, a) for env_, a in zip(envs, actions)]
        wait(futures)

        out = [pack_out.result() for pack_out in futures]

        next_s = [out[_][0].tolist() for _ in range(self.agent_n)]
        r = [out[_][1] for _ in range(self.agent_n)]
        d = [out[_][2] for _ in range(self.agent_n)]
        a = [out[_][3] for _ in range(self.agent_n)]
        return next_s, r, d, a

    def _pool_reset(self, envs):
        def __pool_reset(env, ep):
            env.reset(file_name=f'{ep}')

        calculate_ep = []
        for i in range(self.agent_n):
            self.Wd[i]['ep'] = self.episode
            calculate_ep.append(self.episode)
            self.episode += 1

        futures = [self.pool.submit(__pool_reset, env_, ep_) for env_, ep_ in zip(envs, calculate_ep)]
        wait(futures)
        print('All Env Reset !!')

    def _pool_done_reset(self, envs, dones):
        done_envs = []
        done_envs_ep = []
        for i in range(self.agent_n):
            if dones[i]:
                self.Wd[i]['ep'] = self.episode
                done_envs.append(envs[i])
                done_envs_ep.append(self.episode)
                self.episode += 1

        def __pool_done_reset(env, ep):
            env.reset(file_name=f'{ep}')

        futures = [self.pool.submit(__pool_done_reset, env_, ep_) for env_, ep_ in zip(done_envs, done_envs_ep)]
        wait(futures)

    def _run_exploit(self, net, s):
        with torch.no_grad():
            a = net.get_act(s)
        return a.item()

    def _run_explore(self, net, s):
        with torch.no_grad():
            a, _, _ = net.sample(s)
        return a.item()

    def _run_learn(self, steps, interval_steps):
        return steps % interval_steps == 0

    def _run_update_target(self):
        return

    def _run(self,
             envs, replay_buffer,
             max_episodes, max_steps, interval_steps, target_update_interval, batch_size):
        print('Run' + '=' * 50)
        steps = 0
        self.episode = 0
        self.writer_ep = 0
        # Worker mem
        self.Wd = {i: {'ep_acur': 0, 'ep_q1': 0, 'ep_q2': 0, 'ep_p': 0, 'ep':0} for i in range(self.agent_n)}

        self._pool_reset(envs)
        next_s, r, d, _ = self._pool_one_step(envs, actions=[[0] for _ in range(self.agent_n)])
        s = next_s

        while steps < max_steps and self.episode < max_episodes:
            print(f'Time:[{datetime.now().minute}:{datetime.now().second}]'
                  f'Global_info:[{self.episode}/{max_episodes}][{steps}/{max_steps}]'
                  f'Env_info: {[env_.ENVStep for env_ in envs]}')

            # s 에대한 a 예측
            # a = [self.Actor_Policy_Nets[i].get_act(s[i]) for i in range(self.agent_n)]  # a[0] tensor([[0]])
            a = [[self._run_exploit(self.Actor_Policy_Nets[i], s[i])] for i in range(self.agent_n)] # a[0] [0]

            # CNS Step <-
            next_s, r, d, _ = self._pool_one_step(envs, a)

            # Log
            for i in range(self.agent_n):
                with open(f'./DB_ep_srd/{self.Wd[i]["ep"]}.txt', 'a') as f_ep_srd:
                    f_ep_srd.write(f"{s[i]},{r[i]},{d[i]}\n")

            # Buffer <-
            for s_, a_, r_, next_s_, d_, id in zip(s, a, r, next_s, d, range(self.agent_n)):
                self.Wd[id]['ep_acur'] += r_
                replay_buffer.push(s_, a_, r_, next_s_, d_)

            # s <- next_s
            s = next_s

            # learn
            if replay_buffer.get_length() > batch_size and self._run_learn(steps=steps, interval_steps=interval_steps):
                target_update = True if steps % target_update_interval == 0 else False

                for i in range(self.agent_n):
                    mini_batch = replay_buffer.sample(batch_size, per=False)
                    q1_loss, q2_loss, p_loss = self._update(mini_batch, i, target_update)

                    with open(f'./DB_ep/{self.Wd[i]["ep"]}.txt', 'a') as f:
                        f.write(f"{q1_loss},{q2_loss},{p_loss}\n")

                    self.Wd[i]['ep_q1'] += q1_loss
                    self.Wd[i]['ep_q2'] += q2_loss
                    self.Wd[i]['ep_p'] += p_loss

            # Done ep ??
            for d_, id in zip(d, range(self.agent_n)):
                if d_:
                    print(f"{self.Wd[id]['ep_q1']},{self.Wd[id]['ep_q1']/envs[id].ENVStep},"
                          f"{self.Wd[id]['ep_q2']},{self.Wd[id]['ep_q2']/envs[id].ENVStep},"
                          f"{self.Wd[id]['ep_p']},{self.Wd[id]['ep_p']/envs[id].ENVStep},"
                          f"{self.Wd[id]['ep_acur']},{self.Wd[id]['ep_acur']/envs[id].ENVStep}\n")
                    with open(f'./DB_ep/tot.txt', 'a') as f:
                        f.write(f"{self.Wd[id]['ep']},"
                                f"{self.Wd[id]['ep_q1']},{self.Wd[id]['ep_q1']/envs[id].ENVStep},"
                                f"{self.Wd[id]['ep_q2']},{self.Wd[id]['ep_q2']/envs[id].ENVStep},"
                                f"{self.Wd[id]['ep_p']},{self.Wd[id]['ep_p']/envs[id].ENVStep},"
                                f"{self.Wd[id]['ep_acur']},{self.Wd[id]['ep_acur']/envs[id].ENVStep}\n")

                    self.writer_ep += 1
                    WRITER.add_scalar('Loss/q1', self.Wd[id]['ep_q1'], self.writer_ep)
                    WRITER.add_scalar('Loss/q2', self.Wd[id]['ep_q2'], self.writer_ep)
                    WRITER.add_scalar('Loss/p', self.Wd[id]['ep_p'], self.writer_ep)
                    WRITER.add_scalar('Loss/r', self.Wd[id]['ep_acur'], self.writer_ep)
                    WRITER.add_scalar('Loss-av/q1-av', self.Wd[id]['ep_q1']/envs[id].ENVStep, self.writer_ep)
                    WRITER.add_scalar('Loss-av/q2-av', self.Wd[id]['ep_q2']/envs[id].ENVStep, self.writer_ep)
                    WRITER.add_scalar('Loss-av/p-av', self.Wd[id]['ep_p']/envs[id].ENVStep, self.writer_ep)
                    WRITER.add_scalar('Loss-av/r-av', self.Wd[id]['ep_acur']/envs[id].ENVStep, self.writer_ep)

                    for _ in self.Wd[id].keys():
                        self.Wd[id][_] = 0

            self._pool_done_reset(envs, d)

            steps += self.agent_n

        # End
        print(f'Done Training:'
              f'[{self.episode}/{max_episodes}]'
              f'[{steps}/{max_steps}]' + '=' * 50)


if __name__ == '__main__':
    _ = SAC()
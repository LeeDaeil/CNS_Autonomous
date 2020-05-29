import tensorflow as tf
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
#------------------------------------------------------------------
import threading
import datetime
from collections import deque
import pandas as pd
import numpy as np
from time import sleep
from random import randrange
import matplotlib.pyplot as plt
import os
import shutil
import logging
import logging.handlers
#------------------------------------------------------------------
from Pytorch_A3C.CNS_UDP_FAST import CNS
#------------------------------------------------------------------
get_file_time_path = datetime.datetime.now()
MAKE_FILE_PATH = f'./FAST/VER_1_{get_file_time_path.month}_{get_file_time_path.day}_{get_file_time_path.minute}_{get_file_time_path.second}'
os.mkdir(MAKE_FILE_PATH)
logging.basicConfig(filename='{}/test.log'.format(MAKE_FILE_PATH), format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)
#------------------------------------------------------------------
episode = 0             # Global EP


class MainModel:
    def __init__(self):
        self._make_folder()
        self._make_tensorboaed()
        self.main_net = MainNet(net_type='LSTM', input_pa=6, output_pa=9, time_leg=10)
        #self.main_net.load_model('ROD')

    def run(self):
        worker = self.build_A3C()
        for __ in worker:
            __.start()
            sleep(1)
        print('All agent start done')

        count = 1
        while True:
            sleep(5)
            # 살아 있는지 보여줌
            workers_step = ''
            temp = []
            for i in worker:
                workers_step += '{:3d} '.format(i.db.train_DB['Step'])
                temp.append(i.db.train_DB['Step'])
            print('[{}][max:{:3d}][{}]'.format(datetime.datetime.now(), max(temp), workers_step))
            # 모델 save
            if count == 60:
                self.main_net.save_model('ROD')
                count %= 60
            count += 1

    def build_A3C(self):
        # return: 선언된 worker들을 반환함.
        # 테스트 선택도 여기서 수정할 것
        worker = []
        for cnsip, com_port, max_iter in zip(['192.168.0.9', '192.168.0.7', '192.168.0.4'], [7100, 7200, 7300], [5, 0, 0]):
            if max_iter != 0:
                for i in range(1, max_iter + 1):
                    worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=com_port + i,
                                           CNS_ip=cnsip, CNS_port=com_port + i,
                                           main_net=self.main_net, Sess=self.sess,
                                           Summary_ops=[self.summary_op, self.summary_placeholders,
                                                        self.update_ops, self.summary_writer]))
        return worker

    def _make_tensorboaed(self):
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.summary_placeholders, self.update_ops, self.summary_op = self._setup_summary()
        # tensorboard dir change
        self.summary_writer = tf.summary.FileWriter('{}/a3c'.format(MAKE_FILE_PATH), self.sess.graph)

    def _setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        updata_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, updata_ops, summary_op

    def _make_folder(self):
        fold_list = ['{}/a3c'.format(MAKE_FILE_PATH),
                     '{}/log'.format(MAKE_FILE_PATH),
                     '{}/log/each_log'.format(MAKE_FILE_PATH),
                     '{}/model'.format(MAKE_FILE_PATH),
                     '{}/img'.format(MAKE_FILE_PATH)]
        for __ in fold_list:
            if os.path.isdir(__):
                shutil.rmtree(__)
                sleep(1)
                os.mkdir(__)
            else:
                os.mkdir(__)


class MainNet:
    def __init__(self, net_type='DNN', input_pa=1, output_pa=1, time_leg=1):
        self.net_type = net_type
        self.input_pa = input_pa
        self.output_pa = output_pa
        self.time_leg = time_leg
        self.actor, self.critic = self.build_model(net_type=self.net_type, in_pa=self.input_pa,
                                                   ou_pa=self.output_pa, time_leg=self.time_leg)
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

    def build_model(self, net_type='DNN', in_pa=1, ou_pa=1, time_leg=1):
        # 8 16 32 64 128 256 512 1024 2048
        if net_type == 'DNN':
            state = Input(batch_shape=(None, in_pa))
            shared = Dense(32, input_dim=in_pa, activation='relu', kernel_initializer='glorot_uniform')(state)
            shared = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(shared)
            shared = Dense(70, activation='relu', kernel_initializer='glorot_uniform')(shared)

        elif net_type == 'CNN' or net_type == 'LSTM' or net_type == 'CLSTM':
            state = Input(batch_shape=(None, time_leg, in_pa))
            if net_type == 'CNN':
                shared = Conv1D(filters=10, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=3)(shared)
                shared = Flatten()(shared)
                shared = Dense(64)(shared)
                shared = Dense(70)(shared)

            elif net_type == 'LSTM':
                shared = LSTM(32)(state)
                shared = Dense(64)(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=3)(shared)
                shared = LSTM(12)(shared)
                shared = Dense(24)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        # actor_hidden = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(shared)
        actor_hidden = Dense(24, activation='sigmoid', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        # value_hidden = Dense(32, activation='relu', kernel_initializer='he_uniform')(shared)
        value_hidden = Dense(12, activation='sigmoid', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        print('Make {} Network'.format(net_type))

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary(print_fn=logging.info)
        critic.summary(print_fn=logging.info)

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.output_pa))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        # optimizer = Adam(lr=0.01)
        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.0001)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        # optimizer = Adam(lr=0.01)
        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.0001)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    def save_model(self, name):
        self.actor.save_weights("{}/Model/{}_A3C_actor.h5".format(MAKE_FILE_PATH, name))
        self.critic.save_weights("{}/Model/{}_A3C_cric.h5".format(MAKE_FILE_PATH, name))

    def load_model(self, name):
        self.actor.load_weights("FAST/VER_0_3_12_57_57/Model/{}_A3C_actor.h5".format(name))
        self.critic.load_weights("FAST/VER_0_3_12_57_57/Model/{}_A3C_cric.h5".format(name))


class A3Cagent(threading.Thread):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, main_net, Sess, Summary_ops):
        threading.Thread.__init__(self)
        # CNS와 통신과 데이터 교환이 가능한 모듈 호출
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)

        # 중간 멈추기
        self.save_operation_point = {}

        # 네트워크 정보
        if True:
            # copy main network
            self.main_net = main_net

            # input에 대한 정보 input의 경우 2개의 네트워크 동일하게 공유
            __, self.input_time_length, self.input_para_number = self.main_net.actor.input_shape

        # 훈련 정보를 저장하는 모듈
        if True:
            # Tensorboard
            self.sess = Sess
            [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer] = Summary_ops

            # logger
            self.logger = logging.getLogger('{}'.format(self.name))
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.FileHandler('{}/log/each_log/{}.log'.format(MAKE_FILE_PATH, self.name)))

            # 보상이나 상태를 저장하는 부분
            self.db = DB()
            self.db.initial_train_DB()

        # 사용되는 입력 파라메터 업데이트
        self.save_tick = deque([0, 0], maxlen=2)
        self.save_st = deque([False, False], maxlen=2)
        self.gap = 0
        self.rod_start = False
        self.hold_tick = 60*30  # 60 tick * 30분
        self.end_time = 0

        self.one_agents_episode = 0

        done_, R_, Succes_ = self.update_parameter(A=0)

    def update_parameter(self, A):
        '''
        네트워크에 사용되는 input 및 output에 대한 정보를 세부적으로 작성할 것.
        '''

        def get_v(val):
            return self.CNS.mem[val]['Val']

        # 사용되는 파라메터 전체 업데이트
        self.Time_tick = self.CNS.mem['KCNTOMS']['Val']
        self.Critical = get_v('CRETIV')
        self.MWe_power = get_v('ZINST124')/1000
        self.Char_pump_2 = get_v('KLAMPO70')
        self.BHV22 = get_v('BHV22')
        self.RHR_pump = get_v('KLAMPO55')

        self.state = [
            # 네트워크의 Input 에 들어 가는 변수 들
            self.Time_tick, self.Critical,self.MWe_power,self.Char_pump_2,self.BHV22,self.RHR_pump
            ]

        self.save_state = {key: self.CNS.mem[key]['Val'] for key in ['KCNTOMS', # cns tick
                                                                     'CRETIV', # power
                                                                     'ZINST124', # Tavg
                                                                     'KLAMPO70', # Tref
                                                                     'BHV22', # charging vlave state
                                                                     'KLAMPO55',  # charging vlave state
                                                                     ]}

        # 보상

        Reactivity_control = [0, 0, 0]
        if True:
            # 1) Reactivity
            if self.CNS.mem['CRETIV']['Val'] < 0:
                Reactivity_control[0] = 1
            else:
                Reactivity_control[0] = 0
            # 2) Stabilize or reduce reactor power
            if 0 <= self.CNS.mem['QPROREL']['Val'] < 0.02:
                Reactivity_control[1] += 0.5
            else:
                Reactivity_control[1] += 0
            if self.CNS.mem['ZINST124']['Val'] < 1:
                Reactivity_control[1] += 0.5
            else:
                Reactivity_control[1] += 0
            # 3) Boration addition rate
            if True:
                # 3-1) Charging Line Flow
                if self.CNS.mem['KLAMPO70']['Val'] == 1 and self.CNS.mem['BHV22']['Val'] == 1:
                    Reactivity_control[2] += 0.5
                else:
                    Reactivity_control[2] += 0
                # 3-2) IRWST->HV8->RHR->HV603 Flow
                if self.CNS.mem['KLAMPO55']['Val'] == 1 and self.CNS.mem['ZRWST']['Val'] > 0 \
                        and self.CNS.mem['BHV8']['Val'] == 1 and self.CNS.mem['BHV603']['Val'] >= 1:
                    Reactivity_control[2] += 0.5
                else:
                    Reactivity_control[2] += 0

        done = False
        r = sum(Reactivity_control)

        success = False

        if r == 2.5:
            done = True
            success = True
            r = r / 20 + 0.5
        else:
            r = r / 20

        if self.CNS.mem['KCNTOMS']['Val'] > 120:
            # print('DONE')
            done = True
            success = False
            r = -1
        self.save_state['R'] = r

        return done, r, success

    def run_cns(self, i):
        for _ in range(0, i):
            self.CNS.run_freeze_CNS()

    def predict_action(self, actor, input_window):
        predict_result = actor.predict([[input_window]])
        policy = predict_result[0]
        try:
            action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]
        except:
            print(policy)
            sleep(10000)
        return action, predict_result

    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def send_action(self, action):
        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        if action == 0: self.send_action_append(['KSWO70'], [0])  # Stay
        elif action == 1: self.send_action_append(['KSWO70'], [1])  # Stay
        elif action == 2: pass
        elif action == 3: self.send_action_append(['KSWO81'], [0])  # Stay
        elif action == 4: self.send_action_append(['KSWO81'], [1])  # Stay
        elif action == 5: pass
        elif action == 6: self.send_action_append(['KSWO53'], [0])  # Stay
        elif action == 7: self.send_action_append(['KSWO53'], [1])  # Stay
        elif action == 8: pass

        # 최종 파라메터 전송
        self.CNS._send_control_signal(self.para, self.val)

    def train_network(self):
        def discount_reward(rewards):
            discounted_reward = np.zeros_like(rewards)
            running_add = 0
            for _ in reversed(range(len(rewards))):
                running_add = running_add * 0.99 + rewards[_]
                discounted_reward[_] = running_add
            return discounted_reward
        Dis_reward = discount_reward(self.db.train_DB['Reward'])
        Predicted_values = self.main_net.critic.predict(np.array(self.db.train_DB['S']))
        Advantages = Dis_reward - np.reshape(Predicted_values, len(Predicted_values))

        self.main_net.optimizer[0]([self.db.train_DB['S'], self.db.train_DB['Act'], Advantages])
        self.main_net.optimizer[1]([self.db.train_DB['S'], Dis_reward])

        self.db.initial_each_trian_DB()

    def run(self):
        global episode
        self.cns_speed = 1  # x 배속

        def start_or_initial_cns(mal_time):
            self.db.initial_train_DB()
            self.save_operation_point = {}
            self.CNS.init_cns(initial_nub=1)
            sleep(1)
            self.CNS._send_malfunction_signal(12, 100100, 15)
            sleep(1)
            # self.CNS._send_control_signal(['TDELTA'], [0.2*self.cns_speed])

        iter_cns = 1                    # 반복 - 몇 초마다 Action 을 전송 할 것인가?
        mal_time = randrange(40, 60)    # 40 부터 60초 사이에 Mal function 발생
        start_or_initial_cns(mal_time=mal_time)

        # 훈련 시작하는 부분
        while episode < 50000:
            # 1. input_time_length 까지 데이터 수집 및 Mal function 이후로 동작
            self.one_agents_episode = episode
            start_ep_time = datetime.datetime.now()
            self.logger.info(f'[{datetime.datetime.now()}] Start ep')
            while True:
                self.run_cns(iter_cns)
                done, R, Success = self.update_parameter(A=2)
                self.db.add_now_state(Now_S=self.state)
                # if len(self.db.train_DB['Now_S']) > self.input_time_length and self.Time_tick >= mal_time * 5:
                #     # 네트워크에 사용할 입력 데이터 다 쌓고 + Mal function이 시작하면 제어하도록 설계
                #     break
                if len(self.db.train_DB['Now_S']) > self.input_time_length:
                    # 네트워크에 사용할 입력 데이터 다 쌓이면 제어하도록 설계
                    break

                # 2.2 제어 정보와, 상태에 대한 정보를 저장한다. - 제어 이전의 데이터 세이브
                self.save_state['Act'] = 2
                self.save_state['time'] = self.db.train_DB['Step'] * self.cns_speed
                self.db.save_state(self.save_state)

                self.db.train_DB['Step'] += iter_cns

            # 2. 반복 수행 시작
            while True:
                if True:
                    # 2.1 최근 상태 정보를 토대 Rod 제어 예측
                    old_state = self.db.train_DB['Now_S'][-self.input_time_length:]

                    # 기본적으로 아래와 같이 상태를 추출하면 (time_length, input_para_nub) 형태로 나옴.
                    Action_net, Action_probability = self.predict_action(self.main_net.actor, old_state)
                    self.db.train_DB['Avg_q_max'] += np.max(Action_probability)
                    self.db.train_DB['Avg_max_step'] += 1

                    # 2.2 최근 상태에 대한 액션을 CNS로 전송하고 뿐만아니라 자동 제어 신호도 전송한다.
                    self.send_action(action=Action_net)

                    # 2.2 제어 정보와, 상태에 대한 정보를 저장한다.
                    self.save_state['Act'] = Action_net
                    self.save_state['time'] = self.db.train_DB['Step']*self.cns_speed
                    self.db.save_state(self.save_state)

                    # 2.3 제어에 대하여 CNS 동작 시키고 현재 상태 업데이트한다.
                    self.run_cns(iter_cns)
                    # 2.4 새로운 상태를 업데이트 하고 상태 평가를 진행 한다.
                    done, R, Succes = self.update_parameter(A=Action_net)
                    self.db.add_now_state(Now_S=self.state) # self.state 가 업데이트 된 상태이다. New state
                    # 2.5 평가를 저장한다.
                    self.db.add_train_DB(S=old_state, R=R, A=Action_net)
                    # 2.5 기타 변수를 업데이트 한다.
                    self.db.train_DB['Step'] += iter_cns

                    # 2.6 일정 시간 마다 네트워크를 업데이트 한다. 또는 죽으면 update 한다.
                    if self.db.train_DB['Up_t'] >= self.db.train_DB['Up_t_end'] or done:
                        self.train_network()
                        self.db.train_DB['Up_t'] = 0
                    else:
                        self.db.train_DB['Up_t'] += 1

                # 2.7 done에 도달함.
                if done:
                    self.logger.info(f'[{datetime.datetime.now()}] Training Done - {start_ep_time}~'
                                     f'{datetime.datetime.now()}')
                    episode += 1
                    # tensorboard update
                    stats = [self.db.train_DB['TotR'],
                             self.db.train_DB['Avg_q_max'] / self.db.train_DB['Avg_max_step'],
                             self.db.train_DB['Step']]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode)

                    self.logger.info(f'[{datetime.datetime.now()}] Save img')
                    # if self.db.train_DB['Step'] > -1:
                    #     self.db.draw_img(current_ep=episode)

                    self.save_tick = deque([0, 0], maxlen=2)
                    self.save_st = deque([False, False], maxlen=2)
                    self.gap = 0
                    self.rod_start = False
                    self.hold_tick = 60 * 30  # 60 tick * 30분
                    self.end_time = 0

                    mal_time = randrange(40, 60)  # 40 부터 60초 사이에 Mal function 발생
                    start_or_initial_cns(mal_time=mal_time)
                    self.logger.info(f'[{datetime.datetime.now()}] Episode_done - {start_ep_time}~'
                                     f'{datetime.datetime.now()}')
                    break


class DB:
    def __init__(self):
        self.train_DB = {'Now_S': [], 'S': [], 'Reward': [], 'Act': [],
                         'Tur_R': [], 'Tur_A': [],
                         'TotR': 0, 'Step': 0,
                         'Avg_q_max': 0, 'Avg_max_step': 0,
                         'T_Avg_q_max': 0, 'T_Avg_max_step': 0,
                         'Up_t': 0, 'Up_t_end': 10,
                         'Net_triger': False, 'Net_triger_time': []}
        self.gp_db = pd.DataFrame()

        self.fig = plt.figure(constrained_layout=True, figsize=(10, 9))
        self.gs = self.fig.add_gridspec(21, 3)
        self.axs = [self.fig.add_subplot(self.gs[0:3, :]),  # 1
                    self.fig.add_subplot(self.gs[3:6, :]),  # 2
                    self.fig.add_subplot(self.gs[6:9, :]),  # 3
                    self.fig.add_subplot(self.gs[9:12, :]),  # 4
                    self.fig.add_subplot(self.gs[12:15, :]),  # 5
                    self.fig.add_subplot(self.gs[15:18, :]),  # 6
                    self.fig.add_subplot(self.gs[18:21, :]),  # 7
                    # self.fig.add_subplot(self.gs[21:24, :]),  # 8
                    # self.fig.add_subplot(self.gs[24:27, :]),  # 9
                    ]

    def initial_train_DB(self):
        self.train_DB = {'Now_S': [], 'S': [], 'Reward': [], 'Act': [],
                         'TotR': 0, 'Step': 0,
                         'Avg_q_max': 0, 'Avg_max_step': 0,
                         'Up_t': 0, 'Up_t_end': 60,
                         'Net_triger': False, 'Net_triger_time': []}
        self.gp_db = pd.DataFrame()

    def initial_each_trian_DB(self):
        for _ in ['S', 'Reward', 'Act']:
            self.train_DB[_] = []

    def add_now_state(self, Now_S):
        self.train_DB['Now_S'].append(Now_S)

    def add_train_DB(self, S, R, A):
        self.train_DB['S'].append(S)
        self.train_DB['Reward'].append(R)
        Temp_R_A = np.zeros(9)
        Temp_R_A[A] = 1
        self.train_DB['Act'].append(Temp_R_A)
        self.train_DB['TotR'] += self.train_DB['Reward'][-1]

    def save_state(self, save_data_dict):
        temp = pd.DataFrame()
        for key in save_data_dict.keys():
            temp[key] = [save_data_dict[key]]
        self.gp_db = self.gp_db.append(temp, ignore_index=True)

    def draw_img(self, current_ep):
        for _ in self.axs:
            _.clear()
        #
        self.axs[0].plot(self.gp_db['KCNTOMS'], self.gp_db['QPROREL'], 'g', label='Power')
        self.axs[0].legend(loc=2, fontsize=5)
        self.axs[0].set_ylabel('PZR pressure [%]')
        self.axs[0].grid()
        #
        self.axs[1].plot(self.gp_db['KCNTOMS'], self.gp_db['R'], 'g', label='Reward')
        self.axs[1].legend(loc=2, fontsize=5)
        self.axs[1].set_ylabel('PZR level')
        self.axs[1].grid()
        #
        self.axs[2].plot(self.gp_db['KCNTOMS'], self.gp_db['UAVLEGM'], 'g', label='Average')
        self.axs[2].plot(self.gp_db['KCNTOMS'], self.gp_db['UP_D'], 'g', label='UP_D', color='red', lw=1)
        self.axs[2].plot(self.gp_db['KCNTOMS'], self.gp_db['UP_O'], 'g', label='UP_O', color='blue', lw=1)
        self.axs[2].plot(self.gp_db['KCNTOMS'], self.gp_db['DOWN_D'], 'g', label='DOWN_D', color='red', lw=1)
        self.axs[2].plot(self.gp_db['KCNTOMS'], self.gp_db['DOWN_O'], 'g', label='DOWN_O', color='blue', lw=1)
        self.axs[2].legend(loc=2, fontsize=5)
        self.axs[2].set_ylabel('PZR level')
        self.axs[2].grid()
        #
        self.axs[3].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO20'], 'g', label='Reward')
        self.axs[3].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO21'], 'b', label='Reward')
        self.axs[3].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO22'], 'r', label='Reward')
        self.axs[3].legend(loc=2, fontsize=5)
        self.axs[3].set_ylabel('PZR level')
        self.axs[3].grid()
        #
        self.axs[4].plot(self.gp_db['KCNTOMS'], self.gp_db['TOT_ROD'], 'g', label='ROD_POS')
        self.axs[4].legend(loc=2, fontsize=5)
        self.axs[4].set_ylabel('ROD pos')
        self.axs[4].grid()

        self.axs[5].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO17'], 'g', label='Set')
        self.axs[5].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO18'], 'g', label='Acc')
        self.axs[5].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO19'], 'g', label='Real')
        self.axs[5].legend(loc=2, fontsize=5)
        self.axs[5].set_ylabel('Turbine Real')
        self.axs[5].grid()

        self.axs[6].plot(self.gp_db['KCNTOMS'], self.gp_db['UAVLEGS'], 'g', label='PZR_temp')
        self.axs[6].legend(loc=2, fontsize=5)
        self.axs[6].set_ylabel('PZR temp')
        self.axs[6].grid()
        # #
        # self.axs[3].plot(self.gp_db['time'], self.gp_db['BFV122_pos'], 'g', label='BFV122_POS')
        # self.axs[3].legend(loc=2, fontsize=5)
        # self.axs[3].set_ylabel('BFV122 POS [%]')
        # self.axs[3].grid()
        # #
        # self.axs[4].plot(self.gp_db['time'], self.gp_db['BFV122_close_act'], 'g', label='Close')
        # self.axs[4].plot(self.gp_db['time'], self.gp_db['BFV122_open_act'], 'r', label='Open')
        # self.axs[4].set_ylabel('BFV122 Sig')
        # self.axs[4].legend(loc=2, fontsize=5)
        # self.axs[4].grid()
        # #
        # self.axs[5].plot(self.gp_db['time'], self.gp_db['HV142_pos'], 'r', label='HV142_POS')
        # self.axs[5].set_ylabel('HV142 POS [%]')
        # self.axs[5].legend(loc=2, fontsize=5)
        # self.axs[5].grid()
        # #
        # self.axs[6].plot(self.gp_db['time'], self.gp_db['HV142_close_act'], 'g', label='Close')
        # self.axs[6].plot(self.gp_db['time'], self.gp_db['HV142_open_act'], 'r', label='Open')
        # self.axs[6].set_ylabel('HV142 Sig')
        # self.axs[6].legend(loc=2, fontsize=5)
        # self.axs[6].grid()
        # #
        # self.axs[7].plot(self.gp_db['time'], self.gp_db['Charging_flow'], 'g', label='Charging_flow')
        # self.axs[7].plot(self.gp_db['time'], self.gp_db['Letdown_HX_flow'], 'r', label='Letdown_HX_flow')
        # self.axs[7].set_ylabel('Flow Sig')
        # self.axs[7].legend(loc=2, fontsize=5)
        # self.axs[7].grid()
        # #
        # self.axs[8].plot(self.gp_db['time'], self.gp_db['R'], 'g', label='Reward')
        # self.axs[8].set_ylabel('Rewaed')
        # self.axs[8].legend(loc=2, fontsize=5)
        # self.axs[8].grid()

        self.fig.savefig(fname='{}/img/{}_{}.png'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep), dpi=300,
                         facecolor=None)
        self.gp_db.to_csv('{}/log/{}_{}.csv'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep))


if __name__ == '__main__':
    test = MainModel()
    test.run()
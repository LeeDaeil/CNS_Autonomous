import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
#------------------------------------------------------------------
import socket
import threading
import datetime
import pandas as pd
from struct import unpack, pack
from numpy import shape
import numpy as np
from time import sleep
from collections import deque
#------------------------------------------------------------------
from Abnormal.CNS_UDP import CNS
#------------------------------------------------------------------
import os
import shutil
#------------------------------------------------------------------

MAKE_FILE_PATH = './VER_34_LSTM_1_5'
os.mkdir(MAKE_FILE_PATH)

#------------------------------------------------------------------
import logging
import logging.handlers
logging.basicConfig(filename='{}/test.log'.format(MAKE_FILE_PATH), format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)
#------------------------------------------------------------------
import matplotlib.pyplot as plt
#------------------------------------------------------------------
episode = 0
episode_test = 0
Max_score = 0       # if A3C model get Max_score, A3C model will draw the Max_score grape
FINISH_TRAIN = False
FINISH_TRAIN_CONDITION = 2.00

class MainModel:
    def __init__(self):
        self._make_folder()
        self._make_tensorboaed()
        self.main_net = MainNet(net_type='LSTM', input_pa=6, output_pa=3, time_leg=10)
        self.test = True

    def run(self):
        worker = self.build_A3C(A3C_test=self.test)
        for __ in worker:
            __.start()
            sleep(1)
        print('All agent start done')
        # count = 1
        # while True:
        #     sleep(2)
        #     # 살아 있는지 보여줌
        #     workers_step = ''
        #     share_sate = ''
        #     looking = ''
        #     temp = []
        #     for i in worker:
        #         workers_step += '{:3d} '.format(i.db.train_DB['Step'])
        #         temp.append(i.db.train_DB['Step'])
        #     print('[{}][max:{:3d}][{}]'.format(datetime.datetime.now(), max(temp), workers_step))
        #     # 모델 save
        #     if count == 60:
        #         self.Rod_net.save_model('ROD')
        #         self.Turbine_net.save_model('TUR')
        #         count %= 60
        #     count += 1

    def build_A3C(self, A3C_test=False):
        '''
        A3C의 worker 들을 구축하는 부분
        :param A3C_test: test하는 중인지
        :return: 선언된 worker들을 반환함.
        '''
        worker = []
        if A3C_test:
            for i in range(1, 21):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7200 + i,
                                       CNS_ip='192.168.0.7', CNS_port=7200 + i,
                                       main_net=self.main_net, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer]))
        else:
            for i in range(1, 21):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7100 + i,
                                       CNS_ip='192.168.0.9', CNS_port=7100 + i,
                                       main_net=self.main_net, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer]))
            # CNS2
            for i in range(1, 21):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7200 + i,
                                       CNS_ip='192.168.0.7', CNS_port=7200 + i,
                                       main_net=self.main_net, Sess=self.sess,
                                       Summary_ops=[self.summary_op, self.summary_placeholders,
                                                    self.update_ops, self.summary_writer]))
            # CNS3
            for i in range(1, 21):
                worker.append(A3Cagent(Remote_ip='192.168.0.10', Remote_port=7300 + i,
                                       CNS_ip='192.168.0.4', CNS_port=7300 + i,
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
        episode_rod_reward = tf.Variable(0.)
        episode_tur_reward = tf.Variable(0.)
        episode_R_avg_max_q = tf.Variable(0.)
        episode_T_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Rod_Reward/Episode', episode_rod_reward)
        tf.summary.scalar('Tur_Reward/Episode', episode_tur_reward)
        tf.summary.scalar('R_Average_Max_Prob/Episode', episode_R_avg_max_q)
        tf.summary.scalar('T_Average_Max_Prob/Episode', episode_T_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        summary_vars = [episode_total_reward, episode_rod_reward, episode_tur_reward,
                        episode_R_avg_max_q, episode_T_avg_max_q, episode_duration]
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
                shared = LSTM(32, activation='relu')(state)
                shared = Dense(64)(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=10, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=3)(shared)
                shared = LSTM(32)(shared)
                shared = Dense(60)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        actor_hidden = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(32, activation='relu', kernel_initializer='he_uniform')(shared)
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
        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        # optimizer = Adam(lr=0.01)
        optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    def save_model(self, name):
        self.actor.save_weights("{}/Model/{}_A3C_actor.h5".format(MAKE_FILE_PATH, name))
        self.critic.save_weights("{}/Model/{}_A3C_cric.h5".format(MAKE_FILE_PATH, name))


class A3Cagent(threading.Thread):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, main_net, Sess, Summary_ops):
        threading.Thread.__init__(self)
        # CNS와 통신과 데이터 교환이 가능한 모듈 호출
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)

        # 네트워크 정보
        if True:
            # copy main network
            self.main_net = main_net

            # input에 대한 정보 input의 경우 2개의 네트워크 동일하게 공유
            self.input_time_length = self.main_net.actor.input_shape[1]
            self.input_para_number = self.main_net.actor.input_shape[2]

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
        self.update_parameter()

    def update_parameter(self):

        # 사용되는 파라메터 전체 업데이트
        self.Time_tick = self.CNS.mem['KCNTOMS']['Val']
        self.Reactor_power = self.CNS.mem['QPROREL']['Val']
        self.TRIP_SIG = self.CNS.mem['KLAMPO9']['Val']
        self.charging_valve_state = self.CNS.mem['KLAMPO95']['Val']
        self.main_feed_valve_1_state = self.CNS.mem['KLAMPO147']['Val']
        self.main_feed_valve_2_state = self.CNS.mem['KLAMPO148']['Val']
        self.main_feed_valve_3_state = self.CNS.mem['KLAMPO149']['Val']
        self.vct_level = self.CNS.mem['ZVCT']['Val']
        self.pzr_level = self.CNS.mem['ZINST63']['Val']

        if True:
            # 원자로 출력 값을 사용하여 최대 최소 바운더리의 값을 구함.
            base_condition = self.Time_tick / (30000 / self.operation_mode)
            up_base_condition = (self.Time_tick * 53) / (1470000 / self.operation_mode)
            low_base_condition = (self.Time_tick * 3) / (98000/ self.operation_mode)  # (30000 / self.operation_mode)

            self.stady_condition = base_condition + 0.02

            if self.stady_condition >= 1.00:
                self.stady_condition, self.up_condition, self.low_condition = 1.00, 1.10, 0.90
            else:
                self.up_condition = up_base_condition + 0.04
                self.low_condition = low_base_condition # + 0.01

            self.distance_up = self.up_condition - self.Reactor_power
            self.distance_low = self.Reactor_power - self.low_condition

        self.Turbine_setpoint = self.CNS.mem['KBCDO17']['Val']
        self.Turbine_ac = self.CNS.mem['KBCDO18']['Val']  # Turbine ac condition
        self.Turbine_real = self.CNS.mem['KBCDO19']['Val']
        self.load_set = self.CNS.mem['KBCDO20']['Val']  # Turbine load set point
        self.load_rate = self.CNS.mem['KBCDO21']['Val']  # Turbine load rate
        self.Mwe_power = self.CNS.mem['KBCDO22']['Val']

        self.Netbreak_condition = self.CNS.mem['KLAMPO224']['Val'] # 0 : Off, 1 : On
        self.trip_block = self.CNS.mem['KLAMPO22']['Val']  # Trip block condition 0 : Off, 1 : On

        if self.Netbreak_condition == 1:
            self.db.train_DB['Net_triger'] = True
            if len(self.db.train_DB['Net_triger_time']) == 0:
                self.db.train_DB['Net_triger_time'].append(self.Time_tick)

        if self.db.train_DB['Net_triger']:
            self.tur_stady_condition = (self.Time_tick - self.db.train_DB['Net_triger_time'][0]) * (2 / 25)
            self.tur_low_condition = (self.Time_tick - self.db.train_DB['Net_triger_time'][0]) * (2 / 25) - 50
            if self.tur_low_condition <= 0:
                self.tur_low_condition = 0
            self.tur_hi_condition = (self.Time_tick - self.db.train_DB['Net_triger_time'][0]) * (2 / 25) + 50
        else:
            self.tur_stady_condition, self.tur_low_condition, self.tur_hi_condition = 0, 0, 0

        self.distance_tur_up = self.tur_hi_condition - self.Mwe_power
        self.distance_tur_low = self.Mwe_power - self.tur_low_condition


        self.steam_dump_condition = self.CNS.mem['KLAMPO150']['Val'] # 0: auto 1: man
        self.heat_drain_pump_condition = self.CNS.mem['KLAMPO244']['Val'] # 0: off, 1: on
        self.main_feed_pump_1 = self.CNS.mem['KLAMPO241']['Val'] # 0: off, 1: on
        self.main_feed_pump_2 = self.CNS.mem['KLAMPO242']['Val'] # 0: off, 1: on
        self.main_feed_pump_3 = self.CNS.mem['KLAMPO243']['Val'] # 0: off, 1: on
        self.cond_pump_1 = self.CNS.mem['KLAMPO181']['Val'] # 0: off, 1: on
        self.cond_pump_2 = self.CNS.mem['KLAMPO182']['Val'] # 0: off, 1: on
        self.cond_pump_3 = self.CNS.mem['KLAMPO183']['Val'] # 0: off, 1: on

        self.state =[
            self.Reactor_power, self.distance_up, self.distance_low, self.stady_condition, self.Mwe_power/1000,
            # self.distance_tur_up / 1000, self.distance_tur_low / 1000, self.tur_stady_condition / 1000,
            self.load_set/100
        ]

        self.save_state = {
            'CNS_time': self.Time_tick, 'Reactor_power': self.Reactor_power * 100,
            'Reactor_power_up': self.up_condition * 100, 'Reactor_power_low': self.low_condition * 100,

            'Mwe': self.Mwe_power, 'Mwe_up': self.tur_hi_condition, 'Mwe_low':self.tur_low_condition,
            'Turbine_set': self.Turbine_setpoint, 'Turbine_ac': self.Turbine_ac,
            'Turbine_real': self.Turbine_real, 'Load_set': self.load_set,
            'Load_rate': self.load_rate,

            'Net_break': self.Netbreak_condition, 'Trip_block':self.trip_block,
            'Stem_pump': self.steam_dump_condition, 'Heat_pump': self.heat_drain_pump_condition,
            'MF1': self.main_feed_pump_1, 'MF2': self.main_feed_pump_2, 'MF3': self.main_feed_pump_3,
            'CF1': self.cond_pump_1, 'CF2': self.cond_pump_2, 'CF3': self.cond_pump_3,

            'PZR_level': self.pzr_level, 'VCT_level': self.vct_level,
        }

    def run_cns(self, i):
        for _ in range(0, i):
            self.CNS.run_freeze_CNS()

    def predict_action(self, actor, input_window):
        predict_result = actor.predict([[input_window]])
        policy = predict_result[0]
        action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]
        return action, predict_result

    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def send_action(self, action):

        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []

        # 주급수 및 CVCS 자동
        if self.charging_valve_state == 1:
            self.send_action_append(['KSWO100'], [0])
        if self.main_feed_valve_1_state == 1 or self.main_feed_valve_2_state == 1 or self.main_feed_valve_3_state == 1:
            self.send_action_append(['KSWO171', 'KSWO165', 'KSWO159'], [0, 0, 0])
        self.send_action_append(['KSWO78'], [1])

        # 절차서 구성 순서로 진행
        # 1) 출력이 4% 이상에서 터빈 set point를 맞춘다.
        if self.Reactor_power >= 0.04 and self.Turbine_setpoint != 1800:
            if self.Turbine_setpoint < 1750: # 1780 -> 1872
                self.send_action_append(['KSWO213'], [1])
            elif self.Turbine_setpoint >= 1750:
                self.send_action_append(['KSWO213'], [0])
        # 1) 출력 4% 이상에서 터빈 acc 를 200 이하로 맞춘다.
        if self.Reactor_power >= 0.04 and self.Turbine_ac != 210:
            if self.Turbine_ac < 200:
                self.send_action_append(['KSWO215'], [1])
            elif self.Turbine_ac >= 200:
                self.send_action_append(['KSWO215'], [0])
        # 2) 출력 10% 이상에서는 Trip block 우회한다.
        if self.Reactor_power >= 0.10 and self.trip_block != 1:
            self.send_action_append(['KSWO22', 'KSWO21'], [1, 1])
        # 2) 출력 10% 이상에서는 rate를 50까지 맞춘다.
        if self.Reactor_power >= 0.10 and self.Mwe_power <= 0:
            if self.load_set < 100: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
            if self.load_rate < 50: self.send_action_append(['KSWO227', 'KSWO226'], [1, 0])
            else: self.send_action_append(['KSWO227', 'KSWO226'], [0, 0])

        if 0.10 <= self.Reactor_power < 0.20:
            if self.load_set < 100: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.200 <= self.Reactor_power < 0.300:
            if self.load_set < 200: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.300 <= self.Reactor_power < 0.400:
            if self.load_set < 300: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.400 <= self.Reactor_power < 0.500:
            if self.load_set < 400: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.500 <= self.Reactor_power < 0.600:
            if self.load_set < 500: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.600 <= self.Reactor_power < 0.700:
            if self.load_set < 600: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.700 <= self.Reactor_power < 0.800:
            if self.load_set < 700: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.800 <= self.Reactor_power < 0.850:
            if self.load_set < 800: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])
        if 0.850 <= self.Reactor_power < 1.100:
            if self.load_set < 930: self.send_action_append(['KSWO225', 'KSWO224'], [1, 0]) # 터빈 load를 150 Mwe 까지,
            else: self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])

        # 3) 출력 15% 이상 및 터빈 rpm이 1800이 되면 netbreak 한다.
        if self.Reactor_power >= 0.15 and self.Turbine_real >= 1790 and self.Netbreak_condition != 1:
            self.send_action_append(['KSWO244'], [1])
        # 4) 출력 15% 이상 및 전기 출력이 존재하는 경우, steam dump auto로 전향
        if self.Reactor_power >= 0.15 and self.Mwe_power > 0 and self.steam_dump_condition == 1:
            self.send_action_append(['KSWO176'], [0])
        # 4) 출력 15% 이상 및 전기 출력이 존재하는 경우, heat drain pump on
        if self.Reactor_power >= 0.15 and self.Mwe_power > 0 and self.heat_drain_pump_condition == 0:
            self.send_action_append(['KSWO205'], [1])
        # 5) 출력 20% 이상 및 전기 출력이 190Mwe 이상 인경우
        if self.Reactor_power >= 0.20 and self.Mwe_power >= 190 and self.cond_pump_2 == 0:
            self.send_action_append(['KSWO205'],[1])
        # 6) 출력 40% 이상 및 전기 출력이 380Mwe 이상 인경우
        if self.Reactor_power >= 0.40 and self.Mwe_power >= 380 and self.main_feed_pump_2 == 0:
            self.send_action_append(['KSWO193'], [1])
        # 7) 출력 50% 이상 및 전기 출력이 475Mwe
        if self.Reactor_power >= 0.50 and self.Mwe_power >= 475 and self.cond_pump_3 == 0:
            self.send_action_append(['KSWO206'], [1])
        # 8) 출력 80% 이상 및 전기 출력이 765Mwe
        if self.Reactor_power >= 0.80 and self.Mwe_power >= 765 and self.main_feed_pump_3 == 0:
            self.send_action_append(['KSWO192'], [1])

        # 9) 제어봉 조작 신호를 보내기
        if action == 0: self.send_action_append(['KSWO33', 'KSWO32'], [0, 0])  # Stay
        elif action == 1: self.send_action_append(['KSWO33', 'KSWO32'], [1, 0])  # Out
        elif action == 2: self.send_action_append(['KSWO33', 'KSWO32'], [0, 1])  # In

        # 최종 파라메터 전송
        self.CNS._send_control_signal(self.para, self.val)

    def get_reward_done(self):
        if self.Reactor_power > self.stady_condition:
            Rod_R = self.up_condition - self.Reactor_power
        else:
            Rod_R = self.Reactor_power - self.low_condition

        if self.Netbreak_condition == 1:
            if self.Mwe_power > self.tur_stady_condition:
                Tur_R = (self.tur_hi_condition - self.Mwe_power)/1000
            else:
                Tur_R = (self.Mwe_power - self.tur_low_condition)/1000
        else:
            Tur_R = 0

        if self.db.train_DB['Step'] >= 700 and self.Mwe_power < 1:
            done = True

        return done, Rod_R, Tur_R

    def train_network(self):

        def discount_reward(rewards):
            discounted_reward = np.zeros_like(rewards)
            running_add = 0
            for _ in reversed(range(len(rewards))):
                running_add = running_add * 0.99 + rewards[_]
                discounted_reward[_] = running_add
            return discounted_reward
        Rod_dis_reward = discount_reward(self.db.train_DB['Rod_R'])
        Rod_values = self.main_net.critic.predict(np.array(self.db.train_DB['S']))
        Rod_advantages = Rod_dis_reward - np.reshape(Rod_values, len(Rod_values))

        self.main_net.optimizer[0]([self.db.train_DB['S'], self.db.train_DB['Rod_A'], Rod_advantages])
        self.main_net.optimizer[1]([self.db.train_DB['S'], Rod_dis_reward])

        self.db.initial_each_trian_DB()

    def run(self):

        global episode
        self.CNS.init_cns()
        iter_cns = 2    # 반복

        # 훈련 시작하는 부분
        while episode < 5000: # and self.TRIP_SIG == 0:
            # 1. input_time_length 까지 데이터 수집 (10번)
            for i in range(0, self.input_time_length):
                self.run_cns(iter_cns)
                self.update_parameter()
                self.db.add_now_state(Now_S=self.state)
                self.db.train_DB['Step'] += iter_cns

            # 2. 반복 수행 시작
            while True:
                # 2.1 최근 상태 정보를 토대 Rod 제어 예측
                old_state = self.db.train_DB['Now_S'][-self.input_time_length:]

                # 기본적으로 아래와 같이 상태를 추출하면 (time_length, input_para_nub) 형태로 나옴.
                Rod_A, R_pd_r = self.predict_action(self.main_net.actor, old_state)
                self.db.train_DB['R_Avg_q_max'] += np.max(R_pd_r)
                self.db.train_DB['R_Avg_max_step'] += 1

                # 2.2 최근 상태에 대한 액션을 CNS로 전송하고 뿐만아니라 자동 제어 신호도 전송한다.
                self.send_action(action=Rod_A)

                # 2.2 제어 정보와, 상태에 대한 정보를 저장한다.
                self.save_state['Rod_A'] = -1 if Rod_A == 2 else Rod_A
                self.save_state['time'] = self.db.train_DB['Step']
                self.db.save_state(self.save_state)

                # 2.3 제어에 대하여 CNS 동작 시키고 현재 상태 업데이트한다.
                self.run_cns(iter_cns)
                self.update_parameter()
                self.db.add_now_state(Now_S=self.state) # self.state 가 업데이트 된 상태이다. New state
                # 2.4 새로운 상태에 대한 상태 평가를 시작한다.
                done, Rod_R, Tur_R = self.get_reward_done()
                # 2.5 평가를 저장한다.
                self.db.add_train_DB(S=old_state, R_R=Rod_R, R_A=Rod_A, T_R=Tur_R)
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
                    episode += 1
                    # tensorboard update
                    if self.db.train_DB['T_Avg_q_max'] == 0:
                        out_ = 0
                    else:
                        out_ = self.db.train_DB['T_Avg_q_max'] / self.db.train_DB['T_Avg_max_step']

                    stats = [self.db.train_DB['TotR'], self.db.train_DB['TotR_Rod'], self.db.train_DB['TotR_Tur'],
                             self.db.train_DB['R_Avg_q_max'] / self.db.train_DB['R_Avg_max_step'], out_,
                             self.db.train_DB['Step']]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode)
                    # if self.db.train_DB['Step'] > 2000:
                    #     self.db.draw_img(current_ep=episode)

                    # DB initial
                    self.db.initial_train_DB()
                    self.CNS.init_cns()
                    self.CNS._send_malfunction_signal(12, 100100, 10)
                    sleep(2)
                    break


class DB:
    def __init__(self):
        self.train_DB = {'Now_S': [], 'S': [], 'Rod_R': [], 'Rod_A': [],
                         'Tur_R': [], 'Tur_A': [],
                         'TotR': 0, 'Step': 0,
                         'R_Avg_q_max': 0, 'R_Avg_max_step': 0,
                         'T_Avg_q_max': 0, 'T_Avg_max_step': 0,
                         'Up_t': 0, 'Up_t_end': 60,
                         'Net_triger': False, 'Net_triger_time': []}
        self.gp_db = pd.DataFrame()
        self.fig = plt.figure(constrained_layout=True, figsize=(14, 10))
        self.gs = self.fig.add_gridspec(14, 3)
        self.axs = [self.fig.add_subplot(self.gs[0:3, :]),  # 1
                    self.fig.add_subplot(self.gs[3:6, :]),  # 2
                    self.fig.add_subplot(self.gs[6:7, :]),  # 3
                    self.fig.add_subplot(self.gs[7:8, :]),  # 4
                    self.fig.add_subplot(self.gs[8:10, :]),  # 5
                    self.fig.add_subplot(self.gs[10:12, :]),  # 6
                    self.fig.add_subplot(self.gs[12:14, :]), # 7
                    ]

    def initial_train_DB(self):
        self.train_DB = {'Now_S': [], 'S': [], 'Rod_R': [], 'Rod_A': [],
                         'Tur_R': [], 'Tur_A': [],
                         'TotR': 0, 'Step': 0, 'TotR_Rod': 0, 'TotR_Tur': 0,
                         'R_Avg_q_max': 0, 'R_Avg_max_step': 0,
                         'T_Avg_q_max': 0, 'T_Avg_max_step': 0,
                         'Up_t': 0, 'Up_t_end': 60,
                         'Net_triger': False, 'Net_triger_time': []}
        self.gp_db = pd.DataFrame()

    def initial_each_trian_DB(self):
        for _ in ['S', 'Rod_R', 'Rod_A', 'Tur_R', 'Tur_A']:
            self.train_DB[_] = []

    def add_now_state(self, Now_S):
        self.train_DB['Now_S'].append(Now_S)

    def add_train_DB(self, S, R_R, R_A, T_R, T_A):
        self.train_DB['S'].append(S)
        self.train_DB['Rod_R'].append(R_R)
        Temp_R_A = np.zeros(3)
        Temp_R_A[R_A] = 1
        self.train_DB['Rod_A'].append(Temp_R_A)
        Temp_T_A = np.zeros(3)
        Temp_T_A[T_A] = 1
        self.train_DB['Tur_R'].append(T_R)
        self.train_DB['Tur_A'].append(Temp_T_A)

        self.train_DB['TotR'] += self.train_DB['Rod_R'][-1] + self.train_DB['Tur_R'][-1]
        self.train_DB['TotR_Rod'] += self.train_DB['Rod_R'][-1]
        self.train_DB['TotR_Tur'] += self.train_DB['Tur_R'][-1]

    def save_state(self, save_data_dict):
        temp = pd.DataFrame()
        for key in save_data_dict.keys():
            temp[key] = [save_data_dict[key]]
        self.gp_db = self.gp_db.append(temp, ignore_index=True)

    def draw_img(self, current_ep):
        for _ in self.axs:
            _.clear()
        #
        self.axs[0].plot(self.gp_db['time'], self.gp_db['Reactor_power'], 'g', label='Power')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['Reactor_power_up'], 'b', label='Power_hi_bound')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['Reactor_power_low'], 'r', label='Power_low_bound')
        self.axs[0].legend(loc=2, fontsize=5)
        self.axs[0].set_ylabel('Reactor Power [%]')
        self.axs[0].grid()
        #
        self.axs[1].plot(self.gp_db['time'], self.gp_db['Mwe'], 'g', label='Mwe')
        # self.axs[1].plot(self.gp_db['time'], self.gp_db['Mwe_up'], 'b', label='Mwe_hi_bound')
        # self.axs[1].plot(self.gp_db['time'], self.gp_db['Mwe_low'], 'r', label='Mwe_low_bound')
        self.axs[1].plot(self.gp_db['time'], self.gp_db['Load_set'], 'm', label='Set_point')
        self.axs[1].plot(self.gp_db['time'], self.gp_db['Load_rate'], 'y', label='Load_rate')
        self.axs[1].legend(loc=2, fontsize=5)
        self.axs[1].set_ylabel('Electrical Power [MWe]')
        self.axs[1].grid()
        #
        self.axs[2].plot(self.gp_db['time'], self.gp_db['Turbine_set'], 'r')
        self.axs[2].plot(self.gp_db['time'], self.gp_db['Turbine_real'], 'b')
        self.axs[2].set_yticks((900, 1800))
        self.axs[2].set_yticklabels(('900', '1800'))
        self.axs[2].set_ylabel('Turbine RPM')
        self.axs[2].grid()
        #
        self.axs[3].plot(self.gp_db['time'], self.gp_db['Rod_A'], 'black')
        self.axs[3].set_yticks((-1, 0, 1))
        self.axs[3].set_yticklabels(('In', 'Stay', 'Out'))
        self.axs[3].set_ylabel('Rod Control')
        self.axs[3].grid()
        #
        self.axs[4].plot(self.gp_db['time'], self.gp_db['VCT_level'], 'black')
        # self.axs[4].set_yticks((-1, 0, 1))
        # self.axs[4].set_yticklabels(('In', 'Stay', 'Out'))
        self.axs[4].set_ylabel('VCT_level')
        self.axs[4].grid()
        #
        self.axs[5].plot(self.gp_db['time'], self.gp_db['PZR_level'], 'black')
        # self.axs[4].set_yticks((-1, 0, 1))
        # self.axs[4].set_yticklabels(('In', 'Stay', 'Out'))
        self.axs[5].set_ylabel('PZR_level')
        self.axs[5].grid()
        #
        self.axs[6].plot(self.gp_db['time'], self.gp_db['Net_break'], label='Net break')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['Trip_block'], label='Trip block')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['Stem_pump'], label='Stem dump valve auto')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['Heat_pump'], label='Heat pump')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['MF1'], label='Main Feed Water Pump 1')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['MF2'], label='Main Feed Water Pump 2')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['MF3'], label='Main Feed Water Pump 3')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['CF1'], label='Condensor Pump 1')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['CF2'], label='Condensor Pump 2')
        self.axs[6].plot(self.gp_db['time'], self.gp_db['CF3'], label='Condensor Pump 3')
        self.axs[6].set_yticks((0, 1))
        self.axs[6].set_yticklabels(('Off', 'On'))
        self.axs[6].set_xlabel('Time [s]')
        self.axs[6].legend(loc=7, fontsize=5)
        self.axs[6].grid()
        #
        self.fig.savefig(fname='{}/img/{}_{}.png'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep), dpi=600,
                         facecolor=None)
        self.gp_db.to_csv('{}/log/{}_{}.csv'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep))


if __name__ == '__main__':
    test = MainModel()
    test.run()

    #MainNet(net_type='LSTM', input_pa=6, output_pa=3, time_leg=10)
    #MainNet(net_type='DNN', input_pa=6, output_pa=3, time_leg=10)
    #MainNet(net_type='CNN', input_pa=6, output_pa=3, time_leg=10)
    #MainNet(net_type='CLSTM', input_pa=6, output_pa=3, time_leg=10)

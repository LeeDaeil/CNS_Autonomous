import tensorflow as tf
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
#------------------------------------------------------------------
import threading
import datetime
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
from Start_up_sub.CNS_UDP import CNS
#------------------------------------------------------------------
MAKE_FILE_PATH = './VER_1'
os.mkdir(MAKE_FILE_PATH)
logging.basicConfig(filename='{}/test.log'.format(MAKE_FILE_PATH), format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)
#------------------------------------------------------------------
episode = 0             # Global EP


class MainModel:
    def __init__(self):
        self._make_folder()
        self._make_tensorboaed()
        self.main_net = MainNet(net_type='LSTM', input_pa=13, output_pa=3, time_leg=10)

    def run(self):
        worker = self.build_A3C()
        for __ in worker:
            __.start()
            sleep(1)
        print('All agent start done')

        count = 1
        while True:
            sleep(2)
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
        for cnsip, com_port, max_iter in zip(['192.168.0.9', '192.168.0.7', '192.168.0.4'], [7100, 7200, 7300], [20, 20, 20]):
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
                shared = LSTM(32, activation='relu')(state)
                shared = Dense(64)(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=3)(shared)
                shared = LSTM(64)(shared)
                shared = Dense(120)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        # actor_hidden = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(shared)
        actor_hidden = Dense(124, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        # value_hidden = Dense(32, activation='relu', kernel_initializer='he_uniform')(shared)
        value_hidden = Dense(64, activation='relu', kernel_initializer='he_uniform')(shared)
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


class A3Cagent(threading.Thread):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, main_net, Sess, Summary_ops):
        threading.Thread.__init__(self)
        # 운전 관련 정보 분당 x%
        self.operation_mode = 0.5
        # CNS와 통신과 데이터 교환이 가능한 모듈 호출
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        # 이상 징후 발견 용.
        self.monitoring_time_val = 0

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
        self.update_parameter()

    def update_parameter(self):
        '''
        네트워크에 사용되는 input 및 output에 대한 정보를 세부적으로 작성할 것.
        '''

        # 사용되는 파라메터 전체 업데이트
        self.Time_tick = self.CNS.mem['KCNTOMS']['Val']
        self.Reactor_power = self.CNS.mem['QPROREL']['Val']
        self.TRIP = self.CNS.mem['KRXTRIP']['Val']
        # 온도 고려 2019-11-04
        self.Tref_Tavg = self.CNS.mem['ZINST15']['Val']                     # Tref-Tavg
        self.Tavg = self.CNS.mem['UAVLEGM']['Val']                          # 308.21
        self.Tref = self.CNS.mem['UAVLEGS']['Val']                          # 308.22
        # 제어봉 Pos
        self.rod_pos = [self.CNS.mem[nub_rod]['Val'] for nub_rod in ['KBCDO10', 'KBCDO9', 'KBCDO8', 'KBCDO7']]
        #
        self.charging_valve_state = self.CNS.mem['KLAMPO95']['Val']
        self.main_feed_valve_1_state = self.CNS.mem['KLAMPO147']['Val']
        self.main_feed_valve_2_state = self.CNS.mem['KLAMPO148']['Val']
        self.main_feed_valve_3_state = self.CNS.mem['KLAMPO149']['Val']
        self.vct_level = self.CNS.mem['ZVCT']['Val']
        self.pzr_level = self.CNS.mem['ZINST63']['Val']
        #
        self.Turbine_setpoint = self.CNS.mem['KBCDO17']['Val']
        self.Turbine_ac = self.CNS.mem['KBCDO18']['Val']  # Turbine ac condition
        self.Turbine_real = self.CNS.mem['KBCDO19']['Val']
        self.load_set = self.CNS.mem['KBCDO20']['Val']  # Turbine load set point
        self.load_rate = self.CNS.mem['KBCDO21']['Val']  # Turbine load rate
        self.Mwe_power = self.CNS.mem['KBCDO22']['Val']
        #
        self.Netbreak_condition = self.CNS.mem['KLAMPO224']['Val'] # 0 : Off, 1 : On
        self.trip_block = self.CNS.mem['KLAMPO22']['Val']  # Trip block condition 0 : Off, 1 : On
        #
        self.steam_dump_condition = self.CNS.mem['KLAMPO150']['Val'] # 0: auto 1: man
        self.heat_drain_pump_condition = self.CNS.mem['KLAMPO244']['Val'] # 0: off, 1: on
        self.main_feed_pump_1 = self.CNS.mem['KLAMPO241']['Val'] # 0: off, 1: on
        self.main_feed_pump_2 = self.CNS.mem['KLAMPO242']['Val'] # 0: off, 1: on
        self.main_feed_pump_3 = self.CNS.mem['KLAMPO243']['Val'] # 0: off, 1: on
        self.cond_pump_1 = self.CNS.mem['KLAMPO181']['Val'] # 0: off, 1: on
        self.cond_pump_2 = self.CNS.mem['KLAMPO182']['Val'] # 0: off, 1: on
        self.cond_pump_3 = self.CNS.mem['KLAMPO183']['Val'] # 0: off, 1: on

        self.ax_off = self.CNS.mem['CAXOFF']['Val']

        if True:
            # 평균온도에 기반한 출력 증가 알고리즘
            # (290.2~308.2: 18도 증가) -> ( 2%~100%: 98% 증가 )
            # 18 -> 98 따라서 1%증가시 요구되는 온도 증가량 18/98
            # 1분당 1% 증가시 0.00306 도씩 초당 증가해야함.
            # 2% start_ref_temp = 290.2 매틱 마다 0.00306 씩 증가
            start_2per_temp = 291.97
            self.get_current_t_ref = start_2per_temp + (0.0005) * self.Time_tick

            if self.save_operation_point == {}:
                if self.Reactor_power > 0.3:
                    # 저장 이 필요한 부분!
                    self.save_operation_point['get_current_t_ref'] = self.get_current_t_ref
                    self.save_operation_point['time_tick'] = self.Time_tick
                else:
                    pass # 초기 상태 -> 저장이 필요한 부분 전까지
            else:
                if self.Time_tick > self.save_operation_point['time_tick'] + 1500:
                    self.get_current_t_ref = start_2per_temp + (0.0005) * (self.Time_tick - 1500)
                else:
                    self.get_current_t_ref = self.save_operation_point['get_current_t_ref']

            self.up_dead_band = self.get_current_t_ref + 1
            self.down_dead_band = self.get_current_t_ref - 1
            self.up_operation_band = self.get_current_t_ref + 2
            self.down_operation_band = self.get_current_t_ref - 2

        if self.Netbreak_condition == 1:
            self.db.train_DB['Net_triger'] = True
            if len(self.db.train_DB['Net_triger_time']) == 0:
                self.db.train_DB['Net_triger_time'].append(self.Time_tick)

        self.state =[
            # 네트워크의 Input 에 들어 가는 변수 들
            self.Reactor_power, self.up_dead_band/1000, self.down_dead_band/1000, self.get_current_t_ref/1000, self.Mwe_power/1000,
            self.up_operation_band/1000, self.down_operation_band/1000,
            self.load_set/100, self.Tavg/1000,
            self.rod_pos[0]/1000, self.rod_pos[1]/1000, self.rod_pos[2]/1000, self.rod_pos[3]/1000,
        ]

        self.save_state = {
            # 그래프를 그리기 + 데이터 저장 위해서 필요한 변수들
            'CNS_time': self.Time_tick,
            'get_current_t_ref': self.get_current_t_ref, 'Temp_avg': self.Tavg,
            'up_dead': self.up_dead_band, 'down_dead': self.down_dead_band,
            'up_op': self.up_operation_band, 'down_op': self.down_operation_band,

            'ax_off': self.ax_off,

            'Mwe': self.Mwe_power,
            'Turbine_set': self.Turbine_setpoint, 'Turbine_ac': self.Turbine_ac,
            'Turbine_real': self.Turbine_real, 'Load_set': self.load_set,
            'Load_rate': self.load_rate,

            'Net_break': self.Netbreak_condition, 'Trip_block':self.trip_block,
            'Stem_pump': self.steam_dump_condition, 'Heat_pump': self.heat_drain_pump_condition,
            'MF1': self.main_feed_pump_1, 'MF2': self.main_feed_pump_2, 'MF3': self.main_feed_pump_3,
            'CF1': self.cond_pump_1, 'CF2': self.cond_pump_2, 'CF3': self.cond_pump_3,

            'PZR_level': self.pzr_level, 'VCT_level': self.vct_level,
            'Rod_pos': self.rod_pos,
            'Reactor_power': self.Reactor_power,
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

        # Chargning Valve _ mal
        self.send_action_append(['KSWO100'], [1])

        # 주급수 및 CVCS 자동
        if self.charging_valve_state == 1:
            self.send_action_append(['KSWO100'], [0])
        if self.main_feed_valve_1_state == 1 or self.main_feed_valve_2_state == 1 or self.main_feed_valve_3_state == 1:
            self.send_action_append(['KSWO171', 'KSWO165', 'KSWO159'], [0, 0, 0])
        self.send_action_append(['KSWO78'], [1])

        # 절차서 구성 순서로 진행
        # 1) 출력이 4% 이상에서 터빈 set point를 맞춘다.
        if self.Reactor_power >= 0.04 and self.Turbine_setpoint != 1800:
            if self.Turbine_setpoint < 1790: # 1780 -> 1872
                self.send_action_append(['KSWO213'], [1])
            elif self.Turbine_setpoint >= 1790:
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
            if self.load_rate < 10: self.send_action_append(['KSWO227', 'KSWO226'], [1, 0])
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
            self.send_action_append(['KSWO205'], [1])
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

    def get_reward_done(self, A):
        # 현재 상태에 대한 보상 및 게임 종료 계산

        # 보상 계산 -------
        # 현재 reactor power 가 출력 기준 선보다 높아지거나, 낮아지면 거리만큼의 차가 보상으로 제공
        if self.Tavg > self.get_current_t_ref:
            R = (self.up_operation_band - self.Tavg) / 100
        else:
            R = (self.Tavg - self.down_operation_band) / 100
        # Save_R1은 게임 종료의 Trigger
        Save_R1 = R

        # Dead_band 안에 있으면 추가점
        if self.up_dead_band <= self.Tavg <= self.down_dead_band:
            if self.Tavg > self.get_current_t_ref:
                R += (self.up_dead_band - self.Tavg) / 200
            else:
                R += (self.Tavg - self.down_dead_band) / 200
        else:
            pass
        Save_R2 = R

        # action == 0: Stay     action == 1: Out        action == 2: In
        if A == 0:
            R += 0.001
        else:
            pass
        Save_R3 = R

        if Save_R1 < 0 or self.db.train_DB['Step'] >= 8000:
            done = True
        else:
            done = False

        # 각에피소드 마다 Log
        self.logger.info(f'{self.one_agents_episode:4}-{R:.5f}-{Save_R1:.5f}-{Save_R2:.5f}-{Save_R3:.5f}')
        return done, R

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

        self.cns_speed = 2

        def start_or_initial_cns(mal_time):
            self.db.initial_train_DB()
            self.save_operation_point = {}
            self.CNS.init_cns(initial_nub=17)
            # self.CNS._send_malfunction_signal(12, 10001, mal_time)
            sleep(2)
            self.CNS._send_control_signal(['TDELTA'], [0.2*self.cns_speed])
            sleep(1)

        iter_cns = 2                    # 반복 - 몇 초마다 Action 을 전송 할 것인가?
        mal_time = randrange(40, 60)    # 40 부터 60초 사이에 Mal function 발생
        start_or_initial_cns(mal_time=mal_time)

        # 훈련 시작하는 부분
        while episode < 50000:
            # 1. input_time_length 까지 데이터 수집 및 Mal function 이후로 동작
            self.one_agents_episode = episode
            while True:
                self.run_cns(iter_cns)
                self.update_parameter()
                self.db.add_now_state(Now_S=self.state)
                # if len(self.db.train_DB['Now_S']) > self.input_time_length and self.Time_tick >= mal_time * 5:
                #     # 네트워크에 사용할 입력 데이터 다 쌓고 + Mal function이 시작하면 제어하도록 설계
                #     break
                if len(self.db.train_DB['Now_S']) > self.input_time_length:
                    # 네트워크에 사용할 입력 데이터 다 쌓이면 제어하도록 설계
                    break
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
                    self.update_parameter()
                    self.db.add_now_state(Now_S=self.state) # self.state 가 업데이트 된 상태이다. New state
                    # 2.4 새로운 상태에 대한 상태 평가를 시작한다.
                    done, R = self.get_reward_done(A=Action_net)
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
                    episode += 1
                    # tensorboard update
                    stats = [self.db.train_DB['TotR'],
                             self.db.train_DB['Avg_q_max'] / self.db.train_DB['Avg_max_step'],
                             self.db.train_DB['Step']]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode)

                    if self.db.train_DB['Step'] > 100:
                        self.db.draw_img(current_ep=episode)

                    mal_time = randrange(40, 60)  # 40 부터 60초 사이에 Mal function 발생
                    start_or_initial_cns(mal_time=mal_time)
                    break


class DB:
    def __init__(self):
        self.train_DB = {'Now_S': [], 'S': [], 'Reward': [], 'Act': [],
                         'Tur_R': [], 'Tur_A': [],
                         'TotR': 0, 'Step': 0,
                         'Avg_q_max': 0, 'Avg_max_step': 0,
                         'T_Avg_q_max': 0, 'T_Avg_max_step': 0,
                         'Up_t': 0, 'Up_t_end': 60,
                         'Net_triger': False, 'Net_triger_time': []}
        self.gp_db = pd.DataFrame()

        self.fig = plt.figure(constrained_layout=True, figsize=(22, 10))
        self.gs = self.fig.add_gridspec(25, 3)
        self.axs = [self.fig.add_subplot(self.gs[0:3, :]),  # 1
                    self.fig.add_subplot(self.gs[3:6, :]),  # 2
                    self.fig.add_subplot(self.gs[6:9, :]),  # 3
                    self.fig.add_subplot(self.gs[9:12, :]),  # 4
                    self.fig.add_subplot(self.gs[12:14, :]),  # 5
                    self.fig.add_subplot(self.gs[14:16, :]),  # 6
                    self.fig.add_subplot(self.gs[16:18, :]), # 7
                    self.fig.add_subplot(self.gs[18:22, :]),  # 8
                    self.fig.add_subplot(self.gs[22:25, :]),  # 9
                    # self.fig.add_subplot(self.gs[17:20, :]),  # 9
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
        Temp_R_A = np.zeros(3)
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
        self.axs[0].plot(self.gp_db['time'], self.gp_db['Temp_avg'], 'g', label='Temp_avg')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['get_current_t_ref'], 'g', label='get_current_t_ref')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['down_dead'], 'b', label='down_dead')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['down_op'], 'b', label='down_op')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['up_dead'], 'r', label='up_dead')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['up_op'], 'r', label='up_op')
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
        self.axs[3].plot(self.gp_db['time'], self.gp_db['Act'], 'black')
        self.axs[3].set_yticks((0, 1, 2))
        self.axs[3].set_yticklabels(('Stay', 'Out', 'In'))
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
        self.axs[6].legend(loc=7, fontsize=5)
        self.axs[6].grid()
        #
        self.axs[7].plot(self.gp_db['time'], self.gp_db['ax_off'], label='Temp_avg')
        self.axs[7].legend(loc=7, fontsize=5)
        self.axs[7].grid()
        #
        self.axs[8].plot(self.gp_db['time'], self.gp_db['Reactor_power'], label='Reactor')
        self.axs[8].set_xlabel('Time [s]')
        self.axs[8].grid()
        #
        self.fig.savefig(fname='{}/img/{}_{}.png'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep), dpi=600,
                         facecolor=None)
        self.gp_db.to_csv('{}/log/{}_{}.csv'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep))


if __name__ == '__main__':
    test = MainModel()
    test.run()
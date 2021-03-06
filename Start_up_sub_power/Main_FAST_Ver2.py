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

import os
import shutil
import logging
import logging.handlers
#------------------------------------------------------------------
from Start_up_sub_power.CNS_UDP_FAST import CNS
#------------------------------------------------------------------
from Start_up_sub_power.FAST_UI import show_window
#
get_file_time_path = datetime.datetime.now()
MAKE_FILE_PATH = f'./FAST/VER_3_{get_file_time_path.month}_{get_file_time_path.day}_' \
                 f'{get_file_time_path.hour}_' \
                 f'{get_file_time_path.minute}_' \
                 f'{get_file_time_path.second}_'
os.mkdir(MAKE_FILE_PATH)
logging.basicConfig(filename='{}/test.log'.format(MAKE_FILE_PATH), format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)
#------------------------------------------------------------------
episode = 0             # Global EP
MANUAL = False

class MainModel:
    def __init__(self):
        self._make_folder()
        self._make_tensorboaed()
        self.main_net = MainNet(net_type='LSTM', input_pa=8, output_pa=3, time_leg=15)
        #self.main_net.load_model('ROD')

        self.build_info = {
            'IP_list': ['192.168.0.9', '192.168.0.7', '192.168.0.4'],
            'PORT_list': [7100, 7200, 7300],
        }
        if MANUAL:
            self.build_info['Nub'] = [1, 0, 0]
        else:
            self.build_info['Nub'] = [10, 10, 10]

    def run(self):
        worker = self.build_A3C(build_info=self.build_info)
        for __ in worker:
            __.start()
            sleep(1)
        print('All agent start done')

        count = 1
        if MANUAL:
            window_ = show_window(worker)
            window_.start()
            pass
        else:
            while True:
                sleep(1)
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

    def build_A3C(self, build_info):
        # return: 선언된 worker들을 반환함.
        # 테스트 선택도 여기서 수정할 것
        worker = []
        for cnsip, com_port, max_iter in zip(build_info['IP_list'], build_info['PORT_list'], build_info['Nub']):
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
                shared = Conv1D(filters=15, kernel_size=3, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=3)(shared)
                shared = Flatten()(shared)
                shared = Dense(64)(shared)
                shared = Dense(70)(shared)

            elif net_type == 'LSTM':
                shared = LSTM(16, activation='softsign')(state)
                # shared = LSTM(64, return_sequences=True, activation='softsign')(shared)
                # shared = LSTM(64, activation='softsign')(shared)
                # shared = LSTM(12, return_sequences=True, activation='softsign')(shared)
                # shared = LSTM(12, activation='softsign')(shared)

                # shared = LSTM(64)(shared)
                shared = Dense(128)(shared)

            elif net_type == 'CLSTM':
                shared = Conv1D(filters=15, kernel_size=5, strides=1, padding='same')(state)
                shared = MaxPooling1D(pool_size=3)(shared)
                shared = LSTM(12)(shared)
                shared = Dense(24)(shared)

        # ----------------------------------------------------------------------------------------------------
        # Common output network
        # actor_hidden = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(shared)
        actor_hidden = Dense(256, activation='sigmoid')(shared)
        action_prob = Dense(ou_pa, activation='softmax')(actor_hidden)

        # value_hidden = Dense(32, activation='relu', kernel_initializer='he_uniform')(shared)
        value_hidden = Dense(256, activation='sigmoid')(shared)
        state_value = Dense(1, activation='linear')(value_hidden)

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
        # optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
        optimizer = RMSprop(lr=7e-4, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        # optimizer = Adam(lr=0.01)
        optimizer = RMSprop(lr=7e-4, rho=0.99, epsilon=0.01)
        # optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
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

        # NEW_VER_2 Initial COND
        self.COND_INIT = True
        self.COND_INIT_END_TIME = 0
        self.COND_ALL_ROD_OUT = False
        self.COND_NET_BRK = False
        self.COND_NET_BRK_DIS = 0
        self.COND_AFTER = False
        self.COND_AFTER_TIME = 0

        done_, R_ = self.update_parameter(A=0)

    def Log(self, txt):
        out_txt = f'[{datetime.datetime.now()}][{self.one_agents_episode:4}]'
        out_txt += txt
        try:
            self.logger.info(out_txt)
        except:
            pass

    def update_parameter(self, A, only_val_up=False):
        '''
        네트워크에 사용되는 input 및 output에 대한 정보를 세부적으로 작성할 것.
        '''

        # 사용되는 파라메터 전체 업데이트
        self.Time_tick = self.CNS.mem['KCNTOMS']['Val']
        self.Reactor_power = self.CNS.mem['QPROREL']['Val']         # 0.02
        self.Tavg = self.CNS.mem['UAVLEGM']['Val']  # 308.21
        self.Tref = self.CNS.mem['UAVLEGS']['Val']  # 308.22
        self.rod_pos = [self.CNS.mem[nub_rod]['Val'] for nub_rod in ['KBCDO10', 'KBCDO9', 'KBCDO8', 'KBCDO7']]

        self.charging_valve_state = self.CNS.mem['KLAMPO95']['Val'] # 0(Auto) - 1(Man)
        self.main_feed_valve_1_state = self.CNS.mem['KLAMPO147']['Val']
        self.main_feed_valve_2_state = self.CNS.mem['KLAMPO148']['Val']
        self.main_feed_valve_3_state = self.CNS.mem['KLAMPO149']['Val']
        self.vct_level = self.CNS.mem['ZVCT']['Val']                # 74.45
        self.pzr_level = self.CNS.mem['ZINST63']['Val']             # 34.32
        #
        self.boron_conc = self.CNS.mem['KBCDO16']['Val']
        self.make_up_tank = self.CNS.mem['EDEWT']['Val']
        self.boron_tank = self.CNS.mem['EBOAC']['Val']
        #
        self.Turbine_setpoint = self.CNS.mem['KBCDO17']['Val']
        self.Turbine_ac = self.CNS.mem['KBCDO18']['Val']            # Turbine ac condition
        self.Turbine_real = self.CNS.mem['KBCDO19']['Val']          # 20
        self.load_set = self.CNS.mem['KBCDO20']['Val']              # Turbine load set point
        self.load_rate = self.CNS.mem['KBCDO21']['Val']             # Turbine load rate
        self.Mwe_power = self.CNS.mem['KBCDO22']['Val']             # 0

        self.Netbreak_condition = self.CNS.mem['KLAMPO224']['Val']  # 0 : Off, 1 : On
        self.trip_block = self.CNS.mem['KLAMPO22']['Val']           # Trip block condition 0 : Off, 1 : On
        #
        self.steam_dump_condition = self.CNS.mem['KLAMPO150']['Val'] # 0: auto 1: man
        self.heat_drain_pump_condition = self.CNS.mem['KLAMPO244']['Val'] # 0: off, 1: on
        self.main_feed_pump_1 = self.CNS.mem['KLAMPO241']['Val']     # 0: off, 1: on
        self.main_feed_pump_2 = self.CNS.mem['KLAMPO242']['Val']     # 0: off, 1: on
        self.main_feed_pump_3 = self.CNS.mem['KLAMPO243']['Val']     # 0: off, 1: on
        self.cond_pump_1 = self.CNS.mem['KLAMPO181']['Val']          # 0: off, 1: on
        self.cond_pump_2 = self.CNS.mem['KLAMPO182']['Val']          # 0: off, 1: on
        self.cond_pump_3 = self.CNS.mem['KLAMPO183']['Val']          # 0: off, 1: on

        self.ax_off = self.CNS.mem['CAXOFF']['Val']                  # -0.63

        # 보상 조건 계산
        #   [OUTPUT]
        #   - self.Op_ref_power , self.Op_hi_bound , self.Op_low_bound , self.Op_hi_distance, self.Op_low_distance,
        #     self.R_distance
        #   - self.Op_ref_temp , self.Op_T_hi_bound , self.Op_T_low_bound , self.Op_T_hi_distance, self.Op_T_low_distance,
        #     self.R_T_distance
        if self.COND_INIT:
            # Goal 1.8% ~ 2.2% 사이에서 출력 유지

            # Get Op bound
            self.Op_ref_power = 0.020                                   # 0.020 ~ 0.020
            self.Op_hi_bound = 0.030                                    # 0.030 ~ 0.030
            self.Op_low_bound = 0.010                                   # 0.010 ~ 0.010
            self.Op_ref_temp = 291.7                                    #
            self.Op_T_hi_bound = 291.7 + 10
            self.Op_T_low_bound = 291.7 - 10
            # Get Op distance from current power & temp
            self.Op_hi_distance = self.Op_hi_bound - self.Reactor_power
            self.Op_low_distance = self.Reactor_power - self.Op_low_bound
            self.Op_T_hi_distance = self.Op_T_hi_bound - self.Tavg
            self.Op_T_low_distance = self.Tavg - self.Op_T_low_bound
            # Get Fin distance reward
            self.R_distance = min(self.Op_hi_distance, self.Op_low_distance)
            if self.R_distance <= 0:
                self.R_distance = 0

            self.R_T_distance = min(self.Op_T_hi_distance, self.Op_T_low_distance)
            self.R_T_distance = 0
        elif self.COND_ALL_ROD_OUT:
            # Goal 시간 당 1% 씩 출력 증가

            # Get Op bound
            increse_pow_per = 0.01                                      # 시간당 0.03 -> 3% 증가
            one_tick = increse_pow_per / (60 * 300)                     # 300Tick = 1분 -> 60 * 300 = 1시간
                                                                        # 1Tick 당 증가해야할 Power 계산
            update_tick = self.Time_tick - self.COND_INIT_END_TIME      # 현재 - All rod out 해온 운전 시간 빼기
            self.Op_ref_power = update_tick * one_tick + 0.02           # 0.020 ~ 1.000
            support_up = update_tick * one_tick * 1.2 + 0.02           # 0.020 ~ 1.000
            support_down = update_tick * one_tick * 0.8 + 0.02           # 0.020 ~ 1.000

            # if abs(self.Op_ref_power - support_up) >= 0.05:
            #     support_up = self.Op_ref_power + 0.05
            #     support_down = self.Op_ref_power - 0.05

            self.Op_hi_bound = support_up + 0.02                 # 0.040 ~ 1.020
            self.Op_low_bound = support_down - 0.02                # 0.000 ~ 0.980
            self.Op_ref_temp = 291.7                                    #
            self.Op_T_hi_bound = 291.7 + 10
            self.Op_T_low_bound = 291.7 - 10
            # Get Op distance from current power & temp
            self.Op_hi_distance = self.Op_hi_bound - self.Reactor_power
            self.Op_low_distance = self.Reactor_power - self.Op_low_bound
            self.Op_T_hi_distance = self.Op_T_hi_bound - self.Tavg
            self.Op_T_low_distance = self.Tavg - self.Op_T_low_bound
            # Get Fin distance reward
            self.R_distance = min(self.Op_hi_distance, self.Op_low_distance)
            if self.R_distance <= 0:
                self.R_distance = 0

            self.R_T_distance = min(self.Op_T_hi_distance, self.Op_T_low_distance)
            self.R_T_distance = 0
        elif self.COND_NET_BRK:
            # Goal 시간 당 1% 씩 출력 증가 + Tre/ave 보상 제공

            # Get Op bound
            increse_pow_per = 0.01                                  # 시간당 0.03 -> 3% 증가
            one_tick = increse_pow_per / (60 * 300)                 # 300Tick = 1분 -> 60 * 300 = 1시간
                                                                    # 1Tick 당 증가해야할 Power 계산
            update_tick = self.Time_tick - self.COND_INIT_END_TIME  # 현재 - All rod out 해온 운전 시간 빼기
            self.Op_ref_power = update_tick * one_tick + 0.02       # 0.020 ~ 1.000
            support_up = update_tick * one_tick * 1.2 + 0.02           # 0.020 ~ 1.000
            support_down = update_tick * one_tick * 0.8 + 0.02           # 0.020 ~ 1.000

            # if abs(self.Op_ref_power - support_up) >= 0.05:
            #     support_up = self.Op_ref_power + 0.05
            #     support_down = self.Op_ref_power - 0.05

            self.Op_hi_bound = support_up + 0.02                 # 0.040 ~ 1.020
            self.Op_low_bound = support_down - 0.02                # 0.000 ~ 0.980
            self.Op_ref_temp = self.Tref                            #
            self.Op_T_hi_bound = self.Tref + 10
            self.Op_T_low_bound = self.Tref - 10
            # Get Op distance from current power & temp
            self.Op_hi_distance = self.Op_hi_bound - self.Reactor_power
            self.Op_low_distance = self.Reactor_power - self.Op_low_bound
            self.Op_T_hi_distance = self.Op_T_hi_bound - self.Tavg
            self.Op_T_low_distance = self.Tavg - self.Op_T_low_bound
            # Get Fin distance reward
            self.R_distance = min(self.Op_hi_distance, self.Op_low_distance)
            if self.R_distance <= 0:
                self.R_distance = 0

            self.R_T_distance = min(self.Op_T_hi_distance, self.Op_T_low_distance)
            if self.R_distance <= 0:
                self.R_T_distance = 0
        elif self.COND_AFTER:
            # Goal 출력 유지.

            # Get Op bound
            increse_pow_per = 0.01  # 시간당 0.03 -> 3% 증가
            one_tick = increse_pow_per / (60 * 300)  # 300Tick = 1분 -> 60 * 300 = 1시간
            # 1Tick 당 증가해야할 Power 계산
            update_tick = self.COND_AFTER_TIME - self.COND_INIT_END_TIME  # 현재 - All rod out 해온 운전 시간 빼기
            self.Op_ref_power = update_tick * one_tick + 0.02  # 0.020 ~ 1.000
            self.Op_hi_bound = 0.99 + 0.02 # 0.040 ~ 1.020
            self.Op_low_bound = 0.99 - 0.02  # 0.000 ~ 0.980
            self.Op_ref_temp = self.Tref  #
            self.Op_T_hi_bound = self.Tref + 10
            self.Op_T_low_bound = self.Tref - 10
            # Get Op distance from current power & temp
            self.Op_hi_distance = self.Op_hi_bound - self.Reactor_power
            self.Op_low_distance = self.Reactor_power - self.Op_low_bound
            self.Op_T_hi_distance = self.Op_T_hi_bound - self.Tavg
            self.Op_T_low_distance = self.Tavg - self.Op_T_low_bound
            # Get Fin distance reward
            self.R_distance = min(self.Op_hi_distance, self.Op_low_distance)
            # if self.R_distance <= 0:
            #     self.R_distance = 0

            self.R_T_distance = min(self.Op_T_hi_distance, self.Op_T_low_distance)
            # if self.R_distance <= 0:
            #     self.R_T_distance = 0
        else:
            print('ERROR Reward Calculation STEP!')

        # 보상 계산
        #   [OUTPUT]
        #   - R
        if self.COND_INIT or self.COND_ALL_ROD_OUT:
            R = 0
            R += self.R_distance                    # 0 ~ 0.02
        elif self.COND_NET_BRK or self.COND_AFTER:
            R = 0
            R += self.R_distance                    # 0 ~ 0.02
            # self.R_T_distance : [0 ~ 10]
            # if self.R_T_distance >= 0:              # +- 1도 이내
            #     #R_ = 1 - (10 - self.R_T_distance)   # 0 ~ 1
            R_ = self.R_T_distance              # 0 ~ 10
            R_ = R_ / 100                       # 0 ~ 0.01
            # else:
            #     R_ = -0.001                              # +- 1도 넘음
            R += R_                               # 0 ~ 0.02 + 0 ~ 0.02
        else:
            print('ERROR FIN Reward STEP!')

        # Nan 값 방지.
        if self.Tavg == 0:
            R = 0
        R = round(R, 5)

        # 종료 조건 계산
        done_counter = 0
        done_cause = ''
        if self.Reactor_power <= 0.005: # 0.5퍼
            done_cause += f'_Reactor low power{self.Reactor_power}_'
        if self.rod_pos[0] == 0:
            done_counter += 1
            done_cause += '_Reactor Trip_'
        if self.COND_INIT or self.COND_ALL_ROD_OUT:
            if self.R_distance <= 0:
                R += - 0.1
                done_counter += 1
                done_cause += f'_OutPdis{self.R_distance}_'
            if self.Time_tick >= 285000:
                # R += 0.05
                done_counter += 1
                done_cause += '_SUCCESS_Find_NET_'
        elif self.COND_NET_BRK or self.COND_AFTER:
            if self.R_distance <= 0:
                R += - 0.1
                done_counter += 1
                done_cause += f'_OutPdis{self.R_distance}_'
            if self.R_T_distance <= 0:
                R += - 0.1
                done_counter += 1
                done_cause += f'_OutTdis{self.R_T_distance}_'
            if self.COND_AFTER:
                if self.COND_AFTER_TIME + 30000 <= self.Time_tick:
                    R += 1
                    done_counter += 1
                    done_cause += '_SUCCESS_'
            if self.Time_tick >= 285000:
                # R += 1
                done_counter += 1
                done_cause += '_SUCCESS_Find_NET_'

        else:
            print('ERROR END-Point STEP!')

        # Cond Check
        if self.COND_INIT:
            # Cond Check - 해당 상태의 목적 달성하면 상태변화 및 시간 기록 - 이 부분만 존재
            if self.rod_pos[3] >= 221:      # D 뱅크 최대 인출
                self.COND_INIT = False                      # Change COND !!
                self.COND_ALL_ROD_OUT = True                #
                self.COND_NET_BRK = False                   #
                self.COND_AFTER = False
                self.COND_INIT_END_TIME = self.Time_tick    # Save current tick!
        elif self.COND_ALL_ROD_OUT:
            # Cond Check
            if self.Mwe_power >= 1:                         # 전기 출력 발생
                self.COND_INIT = False                      # Change COND !!
                self.COND_ALL_ROD_OUT = False               #
                self.COND_NET_BRK = True                    #
                self.COND_AFTER = False
                # self.COND_INIT_END_TIME = self.Time_tick    # Save current tick!
        elif self.COND_NET_BRK:
            # Cond Check
            if self.Reactor_power >= 0.98:  # 목표 도달
                self.COND_INIT = False  # Change COND !!
                self.COND_ALL_ROD_OUT = False  #
                self.COND_NET_BRK = False  #
                self.COND_AFTER = True
                self.COND_AFTER_TIME = self.Time_tick    # Save current tick!
        elif self.COND_AFTER:
            pass
        else:
            print('ERROR COND Check')

        if True:
            # 최종 종료 조건 계산
            if done_counter > 0:
                done = True
            else:
                done = False

        # 최종 Net_input 기입
        self.Log(txt=f'[Done-{done_counter}][{done_cause}]')
        self.Log(txt=f'[{self.COND_INIT}, {self.COND_ALL_ROD_OUT}, {self.COND_NET_BRK}, {self.COND_AFTER}]')
        self.Log(txt=f'[{self.COND_INIT_END_TIME}, {self.COND_AFTER_TIME}, {self.COND_NET_BRK_DIS}]')
        self.Log(txt=f'[{R:.5f}-{self.R_distance:.5f}-{self.R_T_distance:.5f}-{self.Time_tick:7}]')

        self.state = [
            # 네트워크의 Input 에 들어 가는 변수 들
            round(self.Reactor_power, 5),                   # 0.02 ~ 1.00
            round(self.Op_hi_distance*10/2, 5),            # 0.00 ~ 0.02 -> 0.0 ~ 1.0
            round(self.Op_low_distance*10/2, 5),           # 0.00 ~ 0.02 -> 0.0 ~ 1.0
            # round(self.Op_hi_bound, 5),                     # 0.00 ~ 1.02
            # round(self.Op_low_bound, 5),                    # 0.00 ~ 0.98
            round(self.Tref/310, 5),                        # 0 ~ 310 -> 0 ~ 1.0
            round(self.Tavg/310, 5),                        # 0 ~ 310 -> 0 ~ 1.0
            round(self.Mwe_power/900, 5),                   # 0 ~ 900 -> 0 ~ 1.0
            round(self.Op_T_hi_bound/610, 5),               # 0 ~ 310 -> 0 ~ 1.0
            round(self.Op_T_low_bound/610, 5),              # 0 ~ 310 -> 0 ~ 1.0
            # round(self.Op_T_hi_distance/10, 5),             # 0 ~ 10 -> 0 ~ 1.0
            # round(self.Op_T_low_distance/10, 5),            # 0 ~ 10 -> 0 ~ 1.0
        ]

        self.save_state = {key: self.CNS.mem[key]['Val'] for key in ['KCNTOMS', # cns tick
                                                                     'QPROREL', # power
                                                                     'UAVLEGM', # Tavg
                                                                     'UAVLEGS', # Tref
                                                                     'KLAMPO95', # charging vlave state
                                                                     'KLAMPO147', 'KLAMPO148', 'KLAMPO149',
                                                                     'ZVCT', 'ZINST63',
                                                                     'KBCDO16',
                                                                     'KBCDO17', 'KBCDO18',
                                                                     'KBCDO19', 'KBCDO20', 'KBCDO21', 'KBCDO22',
                                                                     'KLAMPO224', 'KLAMPO22', 'KLAMPO150', 'KLAMPO244',
                                                                     'KLAMPO241', 'KLAMPO242', 'KLAMPO243', 'KLAMPO181',
                                                                     'KLAMPO182', 'KLAMPO183', 'CAXOFF',
                                                                     'KBCDO10', 'KBCDO9', 'KBCDO8', 'KBCDO7',
                                                                     'FANGLE',
                                                                     'EDEWT', 'EBOAC'
                                                                     ]}
        self.save_state['TOT_ROD'] = self.CNS.mem['KBCDO10']['Val'] + \
                                     self.CNS.mem['KBCDO9']['Val'] + \
                                     self.CNS.mem['KBCDO8']['Val'] + \
                                     self.CNS.mem['KBCDO7']['Val']
        self.save_state['R'] = R
        self.save_state['S'] = self.db.train_DB['Step']
        self.save_state['UP_D'] = self.Op_hi_bound
        self.save_state['DOWN_D'] = self.Op_low_bound
        self.save_state['UP_T_D'] = self.Op_T_hi_bound
        self.save_state['DOWN_T_D'] = self.Op_T_low_bound
        for state_val in range(len(self.state)):
            self.save_state[f'{state_val}'] = self.state[state_val]
        return done, R

    def run_cns(self, i):
        for _ in range(0, i):
            if _ == 0:
                # ACT
                self.CNS.run_freeze_CNS()
            else:
                # pass
                self.CNS.run_freeze_CNS()
                self.update_parameter(A=0, only_val_up=True)
                self.send_action(action=0)

    def predict_action(self, actor, input_window):
        predict_result = actor.predict([[input_window]])
        policy = predict_result[0]
        try:
            action = np.random.choice(np.shape(policy)[0], 1, p=policy)[0]
        except:
            print("ERROR from NET!!")
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

        # 주급수 및 CVCS 자동
        if self.charging_valve_state == 1:
             self.send_action_append(['KSWO100'], [0])

        if self.Reactor_power >= 0.20:
            if self.main_feed_valve_1_state == 1 or self.main_feed_valve_2_state == 1 or self.main_feed_valve_3_state == 1:
                self.send_action_append(['KSWO171', 'KSWO165', 'KSWO159'], [0, 0, 0])

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

            # Turbine Load Rate
            if self.load_rate <= 1: self.send_action_append(['KSWO227', 'KSWO226'], [1, 0])
            else: self.send_action_append(['KSWO227', 'KSWO226'], [0, 0])

        def range_fun(st, end, goal):
            if st <= self.Reactor_power < end:
                if self.load_set < goal:
                    self.send_action_append(['KSWO225', 'KSWO224'], [1, 0])  # 터빈 load를 150 Mwe 까지,
                else:
                    if self.Mwe_power + 2 > goal:
                        self.send_action_append(['KSWO225', 'KSWO224'], [1, 0])  # 터빈 load를 150 Mwe 까지,
                    else:
                        self.send_action_append(['KSWO225', 'KSWO224'], [0, 0])

        range_fun(st=0.05, end=0.10, goal=50)
        range_fun(st=0.10, end=0.15, goal=125)
        range_fun(st=0.15, end=0.20, goal=100)
        range_fun(st=0.20, end=0.25, goal=125)
        range_fun(st=0.25, end=0.30, goal=200)
        range_fun(st=0.30, end=0.35, goal=225)
        range_fun(st=0.35, end=0.40, goal=300)
        range_fun(st=0.40, end=0.45, goal=350)
        range_fun(st=0.45, end=0.50, goal=400)
        range_fun(st=0.50, end=0.55, goal=450)
        range_fun(st=0.55, end=0.60, goal=500)
        range_fun(st=0.60, end=0.65, goal=550)
        range_fun(st=0.65, end=0.70, goal=600)
        range_fun(st=0.70, end=0.75, goal=650)
        range_fun(st=0.75, end=0.80, goal=700)
        range_fun(st=0.80, end=0.85, goal=750)
        range_fun(st=0.85, end=0.90, goal=800)
        range_fun(st=0.90, end=0.95, goal=825)
        range_fun(st=0.95, end=0.100, goal=900)

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
        if self.Reactor_power >= 0.20 and self.Mwe_power >= 1 and self.cond_pump_2 == 0:
            self.send_action_append(['KSWO205'], [1])
        # 6) 출력 40% 이상 및 전기 출력이 380Mwe 이상 인경우
        # if self.Reactor_power >= 0.40 and self.Mwe_power >= 380 and self.main_feed_pump_2 == 0:
        if self.Reactor_power >= 0.40 and self.main_feed_pump_2 == 0:
            self.send_action_append(['KSWO193'], [1])
        # 7) 출력 50% 이상 및 전기 출력이 475Mwe
        # if self.Reactor_power >= 0.50 and self.Mwe_power >= 475 and self.cond_pump_3 == 0:
        if self.Reactor_power >= 0.50 and self.cond_pump_3 == 0:
            self.send_action_append(['KSWO206'], [1])
        # 8) 출력 80% 이상 및 전기 출력이 765Mwe
        # if self.Reactor_power >= 0.80 and self.Mwe_power >= 765 and self.main_feed_pump_3 == 0:
        if self.Reactor_power >= 0.80 and self.main_feed_pump_3 == 0:
            self.send_action_append(['KSWO192'], [1])

        # 9) 제어봉 조작 신호
        if divmod(self.Time_tick, 400)[1] == 0:
            if self.rod_pos[3] > 221:
                self.send_action_append(['KSWO33', 'KSWO32'], [0, 0])  # NO ROD CONTROL
            else:
                self.send_action_append(['KSWO33', 'KSWO32'], [1, 0])  # UP ROD CONTROL
        else:
            self.send_action_append(['KSWO33', 'KSWO32'], [0, 0])  # NO ROD CONTROL

        # 9) 제어봉 조작 신호 및 보론 조작 신호를 보내기
        if self.COND_INIT:
            self.send_action_append(['KSWO75'], [1])
            if action == 0:  # stay pow
                pass
            elif action == 1:  # increase pow
                self.send_action_append(['KSWO33', 'KSWO32'], [1, 0])  # UP ROD CONTROL
            elif action == 2:  # decrease pow
                self.send_action_append(['EBOAC'], [50])  # MAKE-Up
        elif self.COND_ALL_ROD_OUT or self.COND_NET_BRK or self.COND_AFTER:
            if action == 0:  # stay pow
                self.send_action_append(['KSWO75', 'KSWO77'], [1, 0])  # BOR on / ALTDIL off
                self.send_action_append(['WBOAC','WDEWT'], [1, 8])  # Set-Make-up Valve
                self.send_action_append(['EBOAC', 'EDEWT'], [0, 0])  # NO INJECT BORN
            elif action == 1:  # increase pow
                self.send_action_append(['KSWO75', 'KSWO77'], [0, 1])  # BOR off / ALTDIL on
                self.send_action_append(['WBOAC','WDEWT'], [1, 8])     # Valve POS
                # self.send_action_append(['EBOAC', 'EDEWT'], [0, 70])   # MAKE-Up
                self.send_action_append(['EBOAC', 'EDEWT'], [0, 200])   # MAKE-Up
            elif action == 2:  # decrease pow
                self.send_action_append(['KSWO75', 'KSWO77'], [1, 0])  # BOR off / ALTDIL on
                self.send_action_append(['WBOAC','WDEWT'], [1, 8])     # Valve POS
                # self.send_action_append(['EBOAC', 'EDEWT'], [10, 0])   # BORN
                self.send_action_append(['EBOAC', 'EDEWT'], [5, 0])   # BORN
            else:
                print('ERROR ACT')
        else:
            print('ERROR CONTROL PART!!')

        # 최종 파라메터 전송
        # print(self.para)
        self.CNS._send_control_signal(self.para, self.val)
        self.Log(txt=f'SEND ACT\n{self.para}\n{self.val}')

    def train_network(self):

        GAE = True
        if GAE:
            # Generalized advantage estimation 구현
            Dis_reward = []
            #
            v_s_ = self.main_net.critic.predict([[self.db.train_DB['Now_S'][-self.input_time_length:]]])[0][0]
            for r in self.db.train_DB['Reward'][::-1]:
                v_s_ = r + 0.99 * v_s_
                Dis_reward.append(v_s_)
            Dis_reward.reverse()

        else:
            # Typical advantage
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
            # self.CNS.init_cns(initial_nub=17)
            self.CNS.init_cns(initial_nub=20)
            # self.CNS._send_malfunction_signal(12, 10001, mal_time)
            # sleep(1)
            # self.CNS._send_control_signal(['TDELTA'], [0.2*self.cns_speed])
            # sleep(1)

        iter_cns = 1                    # 반복 - 몇 초마다 Action 을 전송 할 것인가?
        mal_time = randrange(40, 60)    # 40 부터 60초 사이에 Mal function 발생
        start_or_initial_cns(mal_time=mal_time)

        # 훈련 시작하는 부분
        while episode < 3000:
            # 1. input_time_length 까지 데이터 수집 및 Mal function 이후로 동작

            # NEW_VER_2 Initial COND
            self.COND_INIT = True
            self.COND_INIT_END_TIME = 0
            self.COND_ALL_ROD_OUT = False
            self.COND_NET_BRK = False
            self.COND_NET_BRK_DIS = 0
            self.COND_AFTER = False
            self.COND_AFTER_TIME = 0

            #
            self.one_agents_episode = episode
            episode += 1

            start_ep_time = datetime.datetime.now()

            # logger
            self.logger = logging.getLogger('{}'.format(self.name))
            self.logger.setLevel(logging.INFO)
            self.logger_hand = logging.FileHandler('{}/log/each_log/{}.log'.format(MAKE_FILE_PATH, self.one_agents_episode))
            self.logger.addHandler(self.logger_hand)
            self.logger.info(f'[{datetime.datetime.now()}] Start ep')

            while True:
                self.run_cns(iter_cns)
                done, R = self.update_parameter(A=0)
                self.db.add_now_state(Now_S=self.state)
                # if len(self.db.train_DB['Now_S']) > self.input_time_length and self.Time_tick >= mal_time * 5:
                #     # 네트워크에 사용할 입력 데이터 다 쌓고 + Mal function이 시작하면 제어하도록 설계
                #     break
                if len(self.db.train_DB['Now_S']) > self.input_time_length:
                    # 네트워크에 사용할 입력 데이터 다 쌓이면 제어하도록 설계
                    break

                # 2.2 제어 정보와, 상태에 대한 정보를 저장한다. - 제어 이전의 데이터 세이브
                self.save_state['Act'] = 0
                self.save_state['P_A_1'] = 0
                self.save_state['P_A_2'] = 0
                self.save_state['P_A_3'] = 0
                self.save_state['time'] = self.db.train_DB['Step'] * self.cns_speed
                self.db.save_state(self.save_state)
                self.db.train_DB['Step'] += 1

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
                    if MANUAL:
                        Action_net = int(input(f"[{self.db.train_DB['Step']}-{self.Time_tick}]Slected ACT:"))
                        self.send_action(action=Action_net)
                    else:
                        self.send_action(action=Action_net)

                    # 2.2 제어 정보와, 상태에 대한 정보를 저장한다.
                    self.save_state['Act'] = Action_net
                    self.save_state['P_A_1'] = Action_probability[0][0]
                    self.save_state['P_A_2'] = Action_probability[0][1]
                    self.save_state['P_A_3'] = Action_probability[0][2]
                    self.save_state['time'] = self.db.train_DB['Step']*self.cns_speed
                    self.db.save_state(self.save_state)

                    # 2.3 제어에 대하여 CNS 동작 시키고 현재 상태 업데이트한다.
                    self.run_cns(iter_cns)
                    # 2.4 새로운 상태를 업데이트 하고 상태 평가를 진행 한다.
                    done, R = self.update_parameter(A=Action_net)
                    self.db.add_now_state(Now_S=self.state) # self.state 가 업데이트 된 상태이다. New state
                    # 2.5 평가를 저장한다.
                    self.db.add_train_DB(S=old_state, R=R, A=Action_net)
                    # 2.5 기타 변수를 업데이트 한다.
                    self.db.train_DB['Step'] += 1

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
                    # tensorboard update
                    stats = [self.db.train_DB['TotR'],
                             self.db.train_DB['Avg_q_max'] / self.db.train_DB['Avg_max_step'],
                             self.db.train_DB['Step']]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode)

                    self.logger.info(f'[{datetime.datetime.now()}] Save img')
                    if self.db.train_DB['Step'] > 50:
                        self.db.draw_img(current_ep=episode)


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
                    # 핸드러 리셋
                    self.logger.removeHandler(self.logger.handlers[0])
                    break


class DB:
    def __init__(self):
        self.train_DB = {'Now_S': [], 'S': [], 'Reward': [], 'Act': [],
                         'Tur_R': [], 'Tur_A': [],
                         'TotR': 0, 'Step': 0,
                         'Avg_q_max': 0, 'Avg_max_step': 0,
                         'T_Avg_q_max': 0, 'T_Avg_max_step': 0,
                         # 'Up_t': 0, 'Up_t_end': 20,
                         'Up_t': 0, 'Up_t_end': 5,
                         'Net_triger': False, 'Net_triger_time': []}
        self.gp_db = pd.DataFrame()

        # self.fig = plt.figure(constrained_layout=True, figsize=(10, 9))
        # self.gs = self.fig.add_gridspec(24, 3)
        # self.axs = [self.fig.add_subplot(self.gs[0:3, :]),  # 1
        #             self.fig.add_subplot(self.gs[3:6, :]),  # 2
        #             self.fig.add_subplot(self.gs[6:9, :]),  # 3
        #             self.fig.add_subplot(self.gs[9:12, :]),  # 4
        #             self.fig.add_subplot(self.gs[12:15, :]),  # 5
        #             self.fig.add_subplot(self.gs[15:18, :]),  # 6
        #             self.fig.add_subplot(self.gs[18:21, :]),  # 7
        #             self.fig.add_subplot(self.gs[21:24, :]),  # 8
        #             # self.fig.add_subplot(self.gs[24:27, :]),  # 9
        #             ]

    def initial_train_DB(self):
        self.train_DB = {'Now_S': [], 'S': [], 'Reward': [], 'Act': [],
                         'TotR': 0, 'Step': 0,
                         'Avg_q_max': 0, 'Avg_max_step': 0,
                         'Up_t': 0, 'Up_t_end': 5,
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
        Temp_R_A = np.zeros(3)                  # <-------------------- AcT
        Temp_R_A[A] = 1
        self.train_DB['Act'].append(Temp_R_A)
        self.train_DB['TotR'] += self.train_DB['Reward'][-1]

    def save_state(self, save_data_dict):
        temp = pd.DataFrame()
        for key in save_data_dict.keys():
            temp[key] = [save_data_dict[key]]
        self.gp_db = self.gp_db.append(temp, ignore_index=True)

    def draw_img(self, current_ep):
        # for _ in self.axs:
        #     _.clear()
        # #
        # self.axs[0].plot(self.gp_db['KCNTOMS'], self.gp_db['QPROREL'], 'g', label='Power')
        # self.axs[0].plot(self.gp_db['KCNTOMS'], self.gp_db['UP_D'], 'r', label='Power_UP')
        # self.axs[0].plot(self.gp_db['KCNTOMS'], self.gp_db['DOWN_D'], 'r', label='Power_DOWN')
        # self.axs[0].legend(loc=2, fontsize=5)
        # self.axs[0].grid()
        # #
        # self.axs[1].plot(self.gp_db['KCNTOMS'], self.gp_db['R'], 'g', label='Reward')
        # self.axs[1].legend(loc=2, fontsize=5)
        # self.axs[1].grid()
        # #
        # self.axs[2].plot(self.gp_db['KCNTOMS'], self.gp_db['UAVLEGM'], 'g', label='Average')
        # self.axs[2].plot(self.gp_db['KCNTOMS'], self.gp_db['UAVLEGS'], 'r', label='Ref', color='red', lw=1)
        # self.axs[2].legend(loc=2, fontsize=5)
        # self.axs[2].grid()
        # #
        # self.axs[3].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO20'], 'g', label='Load Set')
        # self.axs[3].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO21'], 'b', label='Load Rate')
        # self.axs[3].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO22'], 'r', label='Real Power')
        # self.axs[3].legend(loc=2, fontsize=5)
        # self.axs[3].grid()
        # #
        # self.axs[4].plot(self.gp_db['KCNTOMS'], self.gp_db['TOT_ROD'], 'g', label='ROD_POS')
        # self.axs[4].legend(loc=2, fontsize=5)
        # self.axs[4].grid()
        #
        # self.axs[5].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO17'], 'g', label='Set')
        # self.axs[5].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO18'], 'b', label='Acc')
        # self.axs[5].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO19'], 'r', label='Real')
        # self.axs[5].legend(loc=2, fontsize=5)
        # self.axs[5].grid()
        #
        # self.axs[6].plot(self.gp_db['KCNTOMS'], self.gp_db['KBCDO16'], 'g', label='Boron')
        # self.axs[6].legend(loc=2, fontsize=5)
        # self.axs[6].grid()
        #
        # self.axs[7].plot(self.gp_db['KCNTOMS'], self.gp_db['EDEWT'], 'g', label='Boron Tank')
        # self.axs[7].legend(loc=2, fontsize=5)
        # self.axs[7].grid()
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

        # self.fig.savefig(fname='{}/img/{}_{}.png'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep), dpi=300,
        #                  facecolor=None)
        self.gp_db.to_csv('{}/log/{}_{}.csv'.format(MAKE_FILE_PATH, self.train_DB['Step'], current_ep))


if __name__ == '__main__':
    test = MainModel()
    test.run()
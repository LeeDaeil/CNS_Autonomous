import threading
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense
from keras import backend as K

from CNS_gym import gym
from Auto_cont.Auto_run_freeze import Aurorun
from time import sleep


class PowerControl(threading.Thread):
    def __init__(self, mother_mem, clean):
        threading.Thread.__init__(self)
        self.mother_mem = mother_mem
        self.clean = clean

    def run(self):
        Auto_run = Aurorun()

        env = gym(self.mother_mem)

        Auto_run.initial()
        sleep(1)
        agent = DQNAgent(action_size=3)
        for i in range(3):
            self.clean['Sig'] = True
            sleep(1)
            Auto_run.run()

            print('[{}]Start_CNS_process....'.format(self))
            step = 0

            done = False
            while not done:
                state = env.update_state()
                state = np.reshape(state, [1, 20])
                action = agent.get_action(state)
                next_state, reward, done = env.step(action, iter=step, human=False)
                step += 1
            Auto_run.initial()
            sleep(1)

class DQNAgent:
    def __init__(self, action_size):
        self.load_model = True
        # environment settings
        self.state_size = 20
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = 0.1
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 64
        self.train_start = 1000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=10000)
        self.no_op_steps = 30
        # build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./DQN.h5")

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        if np.random.rand() <= self.epsilon:
            ACT = random.randrange(self.action_size)
            with open('Action_log.txt', 'a') as f:
                f.write('{}\n'.format(ACT))
            return ACT
        else:
            q_value = self.model.predict(history)
            ACT = np.argmax(q_value[0])
            with open('Action_log.txt', 'a') as f:
                f.write('{}:{}\n'.format(ACT, q_value))
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size))
        next_history = np.zeros((self.batch_size, self.state_size))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = mini_batch[i][0]
            next_history[i] = mini_batch[i][3]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * target_value[i][np.argmax(value[i])]
                # np.amax(target_value[i]) DQN

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, name):
        self.model.save_weights(name)

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
import tensorflow as tf
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K

class MainNet:
    def __init__(self, net_type='DNN', input_pa=1, output_pa=1, time_leg=1):
        self.net_type = net_type
        self.input_pa = input_pa
        self.output_pa = output_pa
        self.time_leg = time_leg
        self.actor, self.critic = self.build_model(net_type=self.net_type, in_pa=self.input_pa, ou_pa=self.output_pa, time_leg=self.time_leg)
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

    def build_model(self, net_type='DNN', in_pa=1, ou_pa=1, time_leg=1):
        # 8 16 32 64 128 256 512 1024 2048
        state = Input(batch_shape=(None, time_leg, in_pa))
        shared = LSTM(time_leg, activation='relu', return_sequences=True)(state)
        shared = LSTM(time_leg, activation='relu',  return_sequences=True)(shared)
        shared = LSTM(time_leg, activation='relu')(shared)
        # shared = LSTM(time_leg, activation='relu')(state)
        #shared = Dense(64)(shared)

        action_prob = Dense(ou_pa, activation='softmax', kernel_initializer='glorot_uniform')(shared)

        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(shared)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        print('Make {} Network'.format(net_type))

        actor.summary()
        critic.summary()

if __name__ == '__main__':
    MainNet(input_pa=5, output_pa=9, time_leg=10)

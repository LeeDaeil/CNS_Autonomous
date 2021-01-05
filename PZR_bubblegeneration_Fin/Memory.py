"""
Builder: Daeil Lee 2021-01-02

Ref-Code:
    - https://github.com/ku2482/sac-discrete.pytorch
    -
"""
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, net_type='DNN', seq_len=2):
        self.net_type = net_type
        self.seq_len = seq_len
        self.seq_buffer = deque(maxlen=seq_len)
        self.seq_buffer_next = deque(maxlen=seq_len)

        self.capacity = capacity
        self.buffer = []

        self.position = 0
        self.position_seq_len = 0

        self.tot_ep = 0

    def push(self, state, action, reward, next_state, done):
        if self.net_type == 'DNN':
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done, self.position)
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

        else: # LSTN, C_LSTM
            if len(self.seq_buffer) < self.seq_len:
                self.seq_buffer.append(None)
            if len(self.seq_buffer_next) < self.seq_len:
                self.seq_buffer_next.append(None)

            self.seq_buffer.append(state)
            self.seq_buffer_next.append(next_state)

            if self.seq_buffer[0] is not None:
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)

                # Seq_buffer가 차면 저장
                self.buffer[self.position] = (list(self.seq_buffer), action, reward, next_state, done, self.position)
                self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size, per=True):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, pos = map(np.stack, zip(*batch))  # stack for each element

        if per:
            # 선택된 데이터 제거.
            for _ in pos:
                del self.buffer[_]

        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        # Given data
        #   1sec : [0.1, 1], ['on', 'on'],  0.1, False
        #   2sec : [0.2, 2], ['off', 'on'], 0.2, False

        # If get 2 batch about DNN, ..
        # state = [ [0.1, 1], [0.2, 2] ]
        # act = [ ['on', 'on'], ['off', 'on'] ]
        # reward = [ 0.1, 0.2 ]
        # done = [ False, False ]

        # If get 2 batch about LSTM, CLSTM, ..
        # state = [ [[0.1, 1], [0.2, 2]], [[0.2, 2], [.. , ..]] ]
        # act = [ ['on', 'on'], ['off', 'on'] ]
        # reward = [ 0.1, 0.2 ]
        # done = [ False, False ]

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

    def add_ep(self):
        self.tot_ep += 1

    def get_ep(self):
        return self.tot_ep

    def get_info(self):
        print(self.get_length(), '/', self.capacity)
        print(self.buffer)
        return 0


if __name__ == '__main__':
    Buffer = ReplayBuffer(capacity=100, net_type='LSTM')

    Buffer.push(state=[1, 0.1], next_state=[1.1, 0.2], action=[1, 1], reward=0.1, done=False)
    Buffer.get_info()
    Buffer.push(state=[1.1, 0.2], next_state=[1.2, 0.3], action=[1, 0], reward=0.2, done=False)
    Buffer.get_info()
    Buffer.push(state=[1.2, 0.3], next_state=[1.3, 0.4], action=[1, 0], reward=0.3, done=False)
    Buffer.get_info()
    Buffer.push(state=[1.3, 0.4], next_state=[1.4, 0.5], action=[1, 0], reward=0.4, done=False)
    Buffer.get_info()

    print(Buffer.sample(2))
    Buffer.get_info()
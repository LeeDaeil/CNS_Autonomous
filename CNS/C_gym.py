from CNS.UDP import DataShare
from CNS.Auto_cont.Auto_run_freeze import Aurorun
from numpy import shape
from sklearn.preprocessing import Normalizer

class gym:
    def __init__(self):
        self.CNS = DataShare('192.168.0.29', 7000)
        self.prameter_db = self.read_state_DB()

        self.Min_Max = [[0,    0,   0,   0,   0,   0,   0,   0,  0.0, -12.0,  1.0,
                        200.0, 270.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 0.01],
                       [228, 228, 228, 228, 228, 228, 228, 228, 10.0,  -3.0, 20.0,
                        350.0, 330.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 0.20]]
        self.normalizer = Normalizer().fit(self.Min_Max)

    # 1. reset and make current data
    def reset(self):
        self.CNS.reset()
        current_state = []
        self.CNS.update_mem()
        for i in range(shape(self.prameter_db)[0]):
            current_state.append(self.CNS.mem[self.prameter_db[i]]['Val'])
        # normalization
        current_state = self.normalizer.transform([current_state])  # (20,) -> (1,20)
        return current_state

    # 3. step
    def step(self, action, iter = 0, human = True):
        # 3.1 make action

        if human:
            pass
        else:
            if action == 0:
                self.CNS.sc_value(['KSWO33', 'KSWO32'], [1, 0], '192.168.0.11', 7001)
            elif action == 1:
                self.CNS.sc_value(['KSWO33', 'KSWO32'], [0, 0], '192.168.0.11', 7001)
            else:
                self.CNS.sc_value(['KSWO33', 'KSWO32'], [0, 1], '192.168.0.11', 7001)
        next_state = self.update_sate()
        reward, done = self.make_condition(iter)

        return next_state, reward, done

    # (sub 3.1) make reward
    def make_condition(self, iter = 0):
        power = self.CNS.mem['QPROREL']['Val']

        if iter <= 30:
            if power >= 0.025:
                return 1, True
            else:
                return 0, False
        else: # iter = 31
            if power >= 0.025:
                return 1, True
            else: # iter = 31 and power < 0.025
                return -1, True

    # (sub 3.0) read step state
    def update_sate(self):
        current_state = []
        self.CNS.update_mem()
        for i in range(shape(self.prameter_db)[0]):
            current_state.append(self.CNS.mem[self.prameter_db[i]]['Val'])
        # normalization
        current_state = self.normalizer.transform([current_state])  # (20,) -> (1,20)
        return current_state

    # (sub) read state DB
    def read_state_DB(self):
        temp_ = []
        with open('./CNS/Pa.ini', 'r') as f:
        #with open('./Pa.ini', 'r') as f:
            while True:
                line_ = f.readline()
                if line_ == '':
                    break
                temp_.append(line_.split(',')[1])
        return temp_



if __name__ == '__main__':

    # auto mouse start
    Auto_run = Aurorun()

    CNS = gym()

    Auto_run.initial()
    state = CNS.reset()
    print(state)
    Auto_run.run()

    for i in range(0, 20):
        state=CNS.update_sate()
        print(state)

    Auto_run.initial()
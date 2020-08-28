from torch import multiprocessing as mp

from COLLECT_DB.LOCACNS import ENVCNS

import time
import copy


class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.13', '192.168.0.9', '192.168.0.2']
        self.CNS_PORT_LIST = [7100, 7200, 7300]
        self.CNS_NUMBERS = [10, 10, 10]

        self.TimeLeg = 25 * 60

        # TO CNS_UDP_FASE.py
        self.UpdateIterval = 5


        # range_i_ = range(20010, 20100, 10)
        range_i_ = list(range(10010, 10060, 10)) + list(range(20010, 20060, 10)) + list(range(30010, 30060, 10))
        # range_i__ = range(120100, 122100, 100)
        range_i__ = [0]

        range_i = range(0, len(range_i__) * len(range_i_))
        print(len(range_i))

        range_case = [12 for _ in range_i]
        range_opt = []
        range_time = [30 for _ in range_i]

        range_case2 = [0 for _ in range_i]
        range_opt2 = []
        range_time2 = [5 for _ in range_i]

        for case_1_mal in range_i_:
            for case_2_mal in range_i__:
                range_opt.append(case_1_mal)
                range_opt2.append(case_2_mal)

        self.mal_list = {i: {'Case': case, 'Opt': opt, 'Time': time,
                             'Case2': case2, 'Opt2': opt2, 'Time2': time2} for i, case, opt, time, case2, opt2, time2 in zip(range_i, range_case, range_opt, range_time, range_case2, range_opt2, range_time2)
            # 1: {'Case': 0, 'Opt': 0, 'Time': 0}
        }

    def WInfoWarp(self):
        Info = {
            'Iter': 0
        }
        print('초기 Info Share mem로 선언')
        return Info


class Agent(mp.Process):
    def __init__(self, GlobalNet, MEM, CNS_ip, CNS_port, Remote_ip, Remote_port):
        mp.Process.__init__(self)
        # Work info
        self.W = Work_info()
        # CNS

        self.CNS = ENVCNS(Name=self.name, IP=CNS_ip, PORT=CNS_port)
        # self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port, Max_len=self.W.TimeLeg)
        # self.CNS.LoggerPath = 'DB'

        # SharedMem
        self.mem = MEM
        self.LocalMem = copy.deepcopy(self.mem)
        print(f'Make -- {self}')

    def run(self):
        while True:
            # Get iter
            self.CurrentIter = self.mem['Iter']
            self.mem['Iter'] += 1
            # Mal function initial
            # size, maltime = ran.randint(100, 600), ran.randint(30, 100) * 5
            # mal_case = 36

            try:
                # 1: {'Case': 0, 'Opt': 0, 'Time': 0}
                size = self.W.mal_list[self.CurrentIter]['Opt']
                maltime = self.W.mal_list[self.CurrentIter]['Time']
                mal_case = self.W.mal_list[self.CurrentIter]['Case']

                mal_case2 = self.W.mal_list[self.CurrentIter]['Case2']
                mal_opt2 = self.W.mal_list[self.CurrentIter]['Opt2']
                mal_time2 = self.W.mal_list[self.CurrentIter]['Time2']

                file_name = f'{mal_case}_{size}_{maltime}_{mal_case2}_{mal_opt2}_{mal_time2}'
                # CNS initial
                self.CNS.Reset(mal_case=mal_case, mal_opt=size, mal_time=maltime,
                               # mal_case2=mal_case2, mal_opt2=mal_opt2, mal_time2=mal_time2,
                               file_name=file_name)
                time.sleep(1)
                # self.CNS._send_malfunction_signal(Mal_nub=mal_case2, Mal_opt=mal_opt2, Mal_time=mal_time2)
                # time.sleep(2)

                print(f'DONE initial {file_name}')

                while True:
                    for t in range(self.W.TimeLeg + 1):
                        self.CNS.step(0)
                    print('DONE EP')
                    break
            except:
                print('ERROR')
                break
        print('END')


if __name__ == '__main__':
    W_info = Work_info()
    GlobalModel = ''

    # Make shared mem
    MEM = mp.Manager().dict(W_info.WInfoWarp())

    workers = []
    for cnsip, com_port, max_iter in zip(W_info.CNS_IP_LIST, W_info.CNS_PORT_LIST, W_info.CNS_NUMBERS):
        if max_iter != 0:
            for i in range(1, max_iter + 1):
                workers.append(Agent(GlobalNet=GlobalModel,
                                     MEM=MEM,
                                     CNS_ip=cnsip, CNS_port=com_port + i,
                                     Remote_ip=W_info.CURNET_COM_IP, Remote_port=com_port + i))

    [_.start() for _ in workers]
    [_.join() for _ in workers]
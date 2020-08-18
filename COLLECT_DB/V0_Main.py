from torch import multiprocessing as mp

from CNS_UDP_FAST import CNS

import time
import copy


class Work_info:  # 데이터 저장 및 초기 입력 변수 선정
    def __init__(self):
        self.CURNET_COM_IP = '192.168.0.10'
        self.CNS_IP_LIST = ['192.168.0.7', '192.168.0.4', '192.168.0.2']
        self.CNS_PORT_LIST = [7100, 7200, 7300]
        self.CNS_NUMBERS = [10, 0, 10]

        self.TimeLeg = 600

        # TO CNS_UDP_FASE.py
        self.UpdateIterval = 5


        # range_i_ = range(20010, 20100, 10)
        range_i_ = range(121010, 121210, 10)
        # range_i__ = range(120100, 122100, 100)
        range_i__ = [0]

        range_i = range(0, len(range_i__) * len(range_i_))
        print(len(range_i))

        range_case = [52 for _ in range_i]
        range_opt = []
        range_time = [5 for _ in range_i]

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
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port, Max_len=self.W.TimeLeg)
        self.CNS.LoggerPath = 'DB'
        # SharedMem
        self.mem = MEM
        self.LocalMem = copy.deepcopy(self.mem)
        print(f'Make -- {self}')

    # ==============================================================================================================
    # 제어 신호 보내는 파트
    def send_action_append(self, pa, va):
        for _ in range(len(pa)):
            self.para.append(pa[_])
            self.val.append(va[_])

    def send_action(self, act=0):
        # 전송될 변수와 값 저장하는 리스트
        self.para = []
        self.val = []
        # 최종 파라메터 전송
        self.CNS._send_control_signal(self.para, self.val)
    #
    # ==============================================================================================================
    # 입력 출력 값 생성

    def PreProcessing(self):
        pass

    def CNSStep(self):
        self.CNS.run_freeze_CNS()   # CNS에 취득한 값을 메모리에 업데이트
        self.PreProcessing()        # 취득된 값에 기반하여 db_add.txt의 변수명에 해당하는 값을 재처리 및 업데이트
        self.CNS._append_val_to_list()  # 최종 값['Val']를 ['List']에 저장

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
                self.CNS.reset(initial_nub=1, mal=True, mal_case=mal_case, mal_opt=size, mal_time=maltime,
                               # mal_case2=mal_case2, mal_opt2=mal_opt2, mal_time2=mal_time2,
                               file_name=file_name)
                time.sleep(1)
                # self.CNS._send_malfunction_signal(Mal_nub=mal_case2, Mal_opt=mal_opt2, Mal_time=mal_time2)
                # time.sleep(2)

                print(f'DONE initial {file_name}')

                while True:
                    # 초기 제어 Setting 보내기
                    # self.send_action()
                    # time.sleep(1)

                    # Train Mode
                    # Time Leg 만큼 데이터 수집만 수행
                    for t in range(self.W.TimeLeg + 1):
                        self.CNSStep()
                        # Mal_nub, Mal_opt, Mal_time):
                        # if t == 0:
                        #     self.CNS._send_malfunction_signal(Mal_nub=mal_case2, Mal_opt=mal_opt2, Mal_time=mal_time2)
                        #     time.sleep(100)

                    print('DONE EP')
                    break
            except:
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
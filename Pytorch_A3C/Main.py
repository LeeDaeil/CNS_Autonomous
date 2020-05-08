from Pytorch_A3C.Net_Model import Agent_network
from Pytorch_A3C.CNS_UDP_FAST import CNS
import torch
import torch.multiprocessing as mp
import time
import datetime


class Worker(mp.Process):
    def __init__(self, Remote_ip, Remote_port, CNS_ip, CNS_port, Shared_info,
                 G_net, G_OPT, L_net_name):
        super(Worker, self).__init__()
        self.CNS = CNS(self.name, CNS_ip, CNS_port, Remote_ip, Remote_port)
        self.Shared_info = Shared_info
        pass

    def run(self):
        self.CNS.init_cns(initial_nub=17)
        while True:
            self.CNS.run_freeze_CNS()
            self.Shared_info.value = self.CNS.mem['KCNTOMS']['Val']


class Shared_OPT(torch.optim.Adam):
    def __init__(self, Net_para):
        super(Shared_OPT, self).__init__(Net_para, lr=1e-3, betas=(0.95, 0.99), eps=1e-8,
                                         weight_decay=0)
        print(self.param_groups)


if __name__ == '__main__':
    GlobalNet = Agent_network(agent_name="Main", state_dim=5, state_time=12, action_dim=2)
    GlobalNet.share_memory()
    Opt = Shared_OPT(GlobalNet.parameters())

    Shared_info, Shared_info_iter = [], 0

    workers = []
    for cnsip, com_port, max_iter in zip(['192.168.0.9', '192.168.0.7', '192.168.0.4'],
                                         [7100, 7200, 7300], [10, 10, 0]):
        if max_iter != 0:
            for i in range(1, max_iter + 1):
                Shared_info.append(mp.Value('i', 0))
                workers.append(Worker(Remote_ip='192.168.0.10', Remote_port=com_port + i,
                                      CNS_ip=cnsip, CNS_port=com_port + i,
                                      Shared_info=Shared_info[Shared_info_iter],
                                      G_net=GlobalNet, G_OPT=Opt, L_net_name=f"L_net_{i}"
                                      )
                               )
                Shared_info_iter += 1

    for __ in workers:
        __.start()
        # time.sleep(1)
    # LOOP MONITORING
    while True:
        Fin_out = ''
        for _ in Shared_info:
            Fin_out += f"{_.value} | "
        print(Fin_out)
        time.sleep(1)
    [w.join() for w in workers]
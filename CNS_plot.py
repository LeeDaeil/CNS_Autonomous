import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import style
from time import sleep

class DrawPlot(threading.Thread):
    def __init__(self, mother_memory_list, mother_memory_nub):
        threading.Thread.__init__(self)
        self.mother_memory_list = mother_memory_list
        self.mother_memory_nub = mother_memory_nub

        # initial plot condition
        self.fig = plt.figure()
        self.ax = [self.fig.add_subplot(2,1,1), self.fig.add_subplot(2,1,2)]
        style.use('fivethirtyeight')

    def run(self):
        print('{}: Start Plot Tool'.format(self))
        while True:
            # try:
            sleep(5)
                #self.ax[0].plot(self.mother_memory_nub['Nub'], self.mother_memory_list['ZINST65']['Val'])
            self.ax[0].scatter(self.mother_memory_nub['Nub'], self.mother_memory_list['ZINST65']['Val'])
            plt.show()
            print('a')
            # except:
            #     pass
            sleep(1)



    def animate(self):
        for __ in self.ax: __.clear()
        self.ax[0].plot(self.mother_memory_nub, self.mother_memory_list['ZINST65']['Val'], label='PRZ Pre', linewidth=1)
        self.ax[0].legend(loc='upper center', ncol=5, fontsize=10)
        self.ax[0].axhline(y=161.6, ls='--', color='r', linewidth=1)
        self.ax[0].axhline(y=154.7, ls='--', color='r', linewidth=1)
        self.ax[0].set_ylim(152, 164)
        self.ax[0].set_xlabel('time')
        self.ax[0].set_ylabel('value')

        self.ax[1].plot(self.mother_memory_nub, self.mother_memory_list['UCOLEG1']['Val'], label='Loop1 Tcold', linewidth=1)
        self.ax[1].plot(self.mother_memory_nub, self.mother_memory_list['UCOLEG2']['Val'], label='Loop2 Tcold', linewidth=1)
        self.ax[1].plot(self.mother_memory_nub, self.mother_memory_list['UCOLEG3']['Val'], label='Loop3 Tcold', linewidth=1)
        self.ax[1].legend(loc='upper center', ncol=5, fontsize=10)
        self.ax[1].axhline(y=305, ls='--', color='r', linewidth=1)
        self.ax[1].axhline(y=286.7, ls='--', color='r', linewidth=1)
        self.ax[1].set_ylim(285.705, 310)
        self.ax[1].set_xlabel('time')
        self.ax[1].set_ylabel('value')
        self.fig.tight_layout()

    def close_gp(self):
        plt.close('all')





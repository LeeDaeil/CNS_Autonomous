import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


class DrawPlot(multiprocessing.Process):
    def __init__(self, mother_mem):
        multiprocessing.Process.__init__(self)

        self.fig = plt.figure()
        self.ax = [self.fig.add_subplot(2, 1, 1), self.fig.add_subplot(2, 1, 2)]
        style.use('fivethirtyeight')
        self.mother_memory = mother_mem

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.animate, interval=1000)
        plt.show()

    def animate(self, i):
        for __ in self.ax:
            __.clear()
        self.ax[0].plot(self.mother_memory['Nub'], self.mother_memory['List']['ZINST65']['Val'], label='PRZ Pre', linewidth=1)
        self.ax[0].legend(loc='upper center', ncol=5, fontsize=10)
        self.ax[0].axhline(y=161.6, ls='--', color='r', linewidth=1)
        self.ax[0].axhline(y=154.7, ls='--', color='r', linewidth=1)
        self.ax[0].set_ylim(152, 164)
        self.ax[0].set_xlabel('time')
        self.ax[0].set_ylabel('value')

        self.ax[1].plot(self.mother_memory['Nub'], self.mother_memory['List']['UCOLEG1']['Val'], label='Loop1 Tcold', linewidth=1)
        self.ax[1].plot(self.mother_memory['Nub'], self.mother_memory['List']['UCOLEG2']['Val'], label='Loop2 Tcold', linewidth=1)
        self.ax[1].plot(self.mother_memory['Nub'], self.mother_memory['List']['UCOLEG3']['Val'], label='Loop3 Tcold', linewidth=1)
        self.ax[1].legend(loc='upper center', ncol=5, fontsize=10)
        self.ax[1].axhline(y=305, ls='--', color='r', linewidth=1)
        self.ax[1].axhline(y=286.7, ls='--', color='r', linewidth=1)
        self.ax[1].set_ylim(285.705, 310)
        self.ax[1].set_xlabel('time')
        self.ax[1].set_ylabel('value')
        self.fig.tight_layout()

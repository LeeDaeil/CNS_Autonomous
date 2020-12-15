import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.tensorboard import SummaryWriter

path = 'run'
writer = SummaryWriter(path)

fig = plt.figure()
gs = GridSpec(2, 2, fig)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 1]),
]

v_0 = []
v_1 = []

for i in range(0, 20):
    [ax.clear() for ax in axs]

    v_0.append(i)
    v_1.append(i)

    axs[0].plot(v_0)
    axs[1].plot(v_1)

    fig.set_tight_layout(True)
    fig.canvas.draw()

    writer.add_figure('sample', fig, global_step=i)

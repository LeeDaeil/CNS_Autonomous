import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from COMMONTOOL import PTCureve

DB = pd.read_csv('324.txt')


x, y, z, zero = [], [], [], []
PTY, PTX, BotZ, UpZ = [], [], [], []
RateX, RateY, RateZ = [], [], []
SaveTIMETEMP = {'Temp':0, 'Time':0}

for temp, t, pres, co1, co2 in zip(DB['UAVLEG2'].tolist(), DB['KCNTOMS'].tolist(),
                                   DB['ZINST65'].tolist(), DB['KLAMPO6'].tolist(),
                                   DB['KLAMPO9'].tolist()):
    x.append(temp)
    y.append(-t)
    z.append(pres)

    if co1 == 0 and co2 == 1 and t > 1500:
        if SaveTIMETEMP['Time'] == 0:
            SaveTIMETEMP['Time'] = t
            SaveTIMETEMP['Temp'] = temp

        rate = -55 / (60 * 60 * 5)
        get_temp = rate * (t - SaveTIMETEMP['Time']) + SaveTIMETEMP['Temp']
        RateX.append(get_temp)
        RateY.append(-t)
        RateZ.append(0)

    zero.append(0)

Temp = []
UpPres = []
BotPres = []
for _ in range(0, 350):
    uppres, botpres = PTCureve()._get_pres(_)
    Temp.append([_])
    UpPres.append([uppres])
    BotPres.append([botpres])

PTX = np.array(Temp)
BotZ = np.array(BotPres)
UpZ = np.array(UpPres)
PTY = np.array([[0] for _ in range(0, 350)])

PTX = np.hstack([PTX[:, 0:1], Temp])
BotZ = np.hstack([BotZ[:, 0:1], BotPres])
UpZ = np.hstack([UpZ[:, 0:1], UpPres])
PTY = np.hstack([PTY[:, 0:1], np.array([[-t] for _ in range(0, 350)])])

print(np.shape(PTX))

fig = plt.figure()
ax1 = plt.axes(projection='3d')

ax1.plot3D(RateX, RateY, RateZ, color='orange', lw=1.5, ls='--')

ax1.plot3D([170, 0, 0, 170, 170],
           [y[-1], y[-1], 0, 0, y[-1]],
           [29.5, 29.5, 29.5, 29.5, 29.5], color='black', lw=0.5, ls='--')
ax1.plot3D([170, 0, 0, 170, 170],
           [y[-1], y[-1], 0, 0, y[-1]],
           [17, 17, 17, 17, 17], color='black', lw=0.5, ls='--')
ax1.plot3D([170, 170], [y[-1], y[-1]],
           [17, 29.5], color='black', lw=0.5, ls='--')
ax1.plot3D([170, 170], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')
ax1.plot3D([0, 0], [y[-1], y[-1]], [17, 29.5], color='black', lw=0.5, ls='--')
ax1.plot3D([0, 0], [0, 0], [17, 29.5], color='black', lw=0.5, ls='--')

ax1.plot_surface(PTX, PTY, UpZ, rstride=8, cstride=8, alpha=0.15, color='r')
ax1.plot_surface(PTX, PTY, BotZ, rstride=8, cstride=8, alpha=0.15, color='r')
# ax1.scatter(PTX, PTY, BotZ, marker='*')

ax1.plot3D(x, y, z, color='blue', lw=1.5)

# linewidth or lw: float
ax1.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, z[-1]], color='blue', lw=0.5, ls='--')
ax1.plot3D([0, x[-1]], [y[-1], y[-1]], [z[-1], z[-1]], color='blue', lw=0.5, ls='--')
ax1.plot3D([x[-1], x[-1]], [0, y[-1]], [z[-1], z[-1]], color='blue', lw=0.5, ls='--')

# each
ax1.plot3D(x, y, zero, color='black', lw=1, ls='--')  # temp
ax1.plot3D(zero, y, z, color='black', lw=1, ls='--')  # pres
ax1.plot3D(x, zero, z, color='black', lw=1, ls='--')  # PT

# 절대값 처리
ax1.set_yticklabels([int(_) for _ in abs(ax1.get_yticks())])

ax1.set_xlabel('Temperature')
ax1.set_ylabel('Time [Tick]')
ax1.set_zlabel('Pressure')

ax1.set_xlim(0, 350)
ax1.set_zlim(0, 200)

plt.show()
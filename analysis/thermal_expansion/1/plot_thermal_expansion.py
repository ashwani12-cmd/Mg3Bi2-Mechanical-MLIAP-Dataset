import matplotlib.pyplot as plt
import numpy as np
from ase.lattice.cubic import Diamond
from gpyumd.atoms import GpumdAtoms
from gpyumd.io import read_gpumd
import gpyumd.keyword as kwd
from gpyumd.load import load_thermo
from gpyumd.sim import Simulation


aw = 2
fs = 16
font = {'size'   : fs}
plt.rc('font', **font)
plt.rc('axes' , linewidth=aw)

def set_fig_properties(ax_list):
    ax_list = ax_list if isinstance(ax_list, list) else [ax_list]
    tl = 8
    tw = 2
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in', right=True, top=True)



thermo = load_thermo()
print("Thermo quantities:", list(thermo.keys()))


time = 0.01*np.arange(1,thermo['temperature'].shape[0]+1) # [ps]
NC = 10  # Number of cells in each direction
NT = 10  # Number of temperature steps
temp = np.arange(100,1001,100)
M = thermo['temperature'].shape[0]//NT
a = (thermo['Lx']+thermo['Ly']+thermo['Lz'])/(3*NC)
Pave = (thermo['Px']+thermo['Py']+thermo['Pz'])/3.
a_ave = a.reshape(NT, M)[:,M//2+1:].mean(axis=1)
fit = np.poly1d(np.polyfit(temp, a_ave, deg=1))


axes = list()
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
axes.append(plt.gca())
plt.plot(time, thermo['temperature'])
plt.xlim([0, 200])
plt.gca().set_xticks(range(0,201,50))
plt.ylim([0, 1100])
plt.gca().set_yticks(range(0,1101,500))
plt.ylabel('Temperature (K)')
plt.xlabel('Time (ps)')
plt.title('(a)')

plt.subplot(2,2,2)
axes.append(plt.gca())
plt.plot(time, Pave)
plt.xlim([0, 200])
plt.gca().set_xticks(range(0,201,50))
plt.ylim([-0.1, 0.4])
plt.gca().set_yticks(np.arange(-1,5)/10)
plt.ylabel('Pressure (GPa)')
plt.xlabel('Time (ps)')
plt.title('(b)')

plt.subplot(2,2,3)
axes.append(plt.gca())
plt.plot(time, a,linewidth=3)
plt.xlim([0, 200])
plt.gca().set_xticks(range(0,201,50))
plt.ylim([5.43, 5.48])
plt.gca().set_yticks([5.44,5.46,5.48])
plt.ylabel(r'a ($\AA$)')
plt.xlabel('Time (ps)')
plt.title('(c)')

plt.subplot(2,2,4)
axes.append(plt.gca())
Tpoly = [0, 1100]
plt.plot(Tpoly, fit(Tpoly),color='C3')
plt.scatter(temp, a_ave,s=200,zorder=100,facecolor='none',edgecolors='C0',linewidths=3)
plt.xlim([0, 1100])
plt.gca().set_xticks(range(0,1101,500))
plt.ylim([5.43, 5.48])
plt.gca().set_yticks([5.44,5.46,5.48])
plt.ylabel(r'a ($\AA$)')
plt.xlabel('Temperature (K)')
plt.title('(d)')

set_fig_properties(axes)
plt.tight_layout()
plt.show()

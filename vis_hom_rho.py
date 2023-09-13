import numpy as np
import matplotlib.pyplot as plt
import json, argparse, os
from matplotlib.widgets import Slider, TextBox

parser = argparse.ArgumentParser()
parser.add_argument('-g','--g', help='gamma star, up to 2 decimal places')
parser.add_argument('-rs','--rs', help='rho_seed/rho_in value')
parser.add_argument('-pi','--pi', help='pi, coupling to nematic')
parser.add_argument('-r','--r', help='run number')
#parser.add_argument('-ts','--timestamp', help='timestamp')
args = parser.parse_args()

savedir     = "hom_rho/gammas_{}_rhoseed_{}_pi_{}/run_{}/".format(args.g, args.rs, args.pi, args.r)


with open(savedir+'parameters.json') as jsonFile:
    parameters = json.load(jsonFile)

run       = parameters["run"]
T         = parameters["T"]        # final time
n_steps   = np.int32(parameters["n_steps"])  # number of time steps
K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
Gammas    = parameters["Gammas"]   # rate of Q alignment with mol field H
Pi        = parameters["Pi"]       # strength of alignment
rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
rhoseed   = parameters["rhoseed"]  /rho_in   # seeding density, normalised by 100 mm^-2
rhoisoend = parameters["rhoisoend"]/rho_in   # jamming density
rhonemend = parameters["rhonemend"]/rho_in   # jamming density max for nematic substrate
mx        = parameters["mx"]       # number of points in meshgrid
my        = parameters["my"]       # number of points in meshgrid
n_steps   = 300
dt        = T / n_steps            # time step size

Qxx = np.fromfile(savedir+'Qxx.dat').reshape(n_steps+1, mx, my)
rho    = np.fromfile(savedir+'rho.dat').reshape(n_steps+1, mx, my)
Qxy  = np.fromfile(savedir+'Qxy.dat').reshape(n_steps+1, mx, my)

S_sqrd = 2*(np.square(Qxx)+np.square(Qxy))
theta   = 0.5*np.arctan2(Qxy, Qxx)
nx      = np.cos(theta)
ny      = np.sin(theta)

#setup a meshgrid

tol = 0.001

x   = np.linspace(0+tol, 1-tol, mx)
y   = np.linspace(0+tol, 1-tol, my)
xv, yv  = np.meshgrid(x,y)

times = np.arange(0, T+dt, dt)

fig, ax= plt.subplots(figsize=(8, 6), ncols=1)

#plotfield = rho * S_sqrd
plotfield = np.sqrt(S_sqrd)
fmax = np.max(plotfield)

i=0
sfield = ax.imshow(plotfield[i].T,cmap='plasma', origin='lower', vmin=0, vmax=fmax)
parrow = ax.quiver(49*xv, 49*yv, nx[i], ny[i], color='w')# color='#b3b3b3')
cb     = fig.colorbar(sfield)#,  anchor=(0, 0.3), shrink=0.7)


sax = fig.add_axes([0.1,0.94,0.85,0.02])
tbax = fig.add_axes([0.05, 0.93, 0.04, 0.04])
tb = TextBox(tbax, 'time')
sl = Slider(sax, '', min(times), max(times), valinit=5+min(times))

def plt_snapshot(val):
	val = (abs(times-val)).argmin()
	#print(val, n_steps)
	sfield.set_data(plotfield[val].T)
	parrow.set_UVC(nx[val], ny[val])
	fig.canvas.draw_idle()
	#return (sfield, parrow)
sl.on_changed(plt_snapshot)
#from matplotlib.animation import FuncAnimation
#anim = FuncAnimation(fig, plt_snapshot, frames=300, interval=1, repeat=True)
plt.show()

import ufl
from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from ufl import Measure, grad, inner
import numpy as np
import os, json

with open('paramsnew.json') as jsonFile:
    parameters = json.load(jsonFile)

run       = parameters["run"]
T         = parameters["T"]        # final time
n_steps   = parameters["n_steps"]  # number of time steps
K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
Gammas    = parameters["Gammas"]   # rate of Q alignment with mol field H
Pi        = parameters["Pi"]       # strength of alignment
rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
rhoseed   = parameters["rhoseed"] /rho_in     # seeding density, normalised by 100 mm^-2
rhoisoend = parameters["rhoisoend"] /rho_in   # jamming density
rhonemend = parameters["rhonemend"] /rho_in   # jamming density max for nematic substrate
mx        = np.int32(parameters["mx"])
my        = np.int32(parameters["my"])

n_steps   = 300
dt        = T / n_steps     # time step size

savedir     = "hom_rho/gammas_{:.2f}_rhoseed_{:.2f}_pi_{:.2f}/run_{}/".format(Gammas, rhoseed, Pi, run)
if not os.path.isdir(savedir):
    os.makedirs(savedir)
else:
    run+=1
    savedir = "hom_rho/gammas_{:.2f}_rhoseed_{:.2f}_pi_{:.2f}/run_{}/".format(Gammas, rhoseed, Pi, run)
    os.makedirs(savedir)

#create mesh, define function spaces
mymesh  = mesh.create_unit_square(MPI.COMM_WORLD, mx, my, mesh.CellType.triangle)
dx      = ufl.Measure("dx", domain=mymesh)
P1      = ufl.FiniteElement("Lagrange", mymesh.ufl_cell(), degree=1)
V       = fem.FunctionSpace(mymesh, ufl.MixedElement(P1, P1, P1))
X       = ufl.SpatialCoordinate(mymesh)

# Define test functions (no trials for nonlinear eqtns)
v1, v2, v3 = ufl.TestFunctions(V)

# Define functions for solutions at previous and current time steps
u_np1   = fem.Function(V)   #unknown Q, and density fields
u_n     = fem.Function(V)   #solution from prev step

# Split mixed functions
u1_np1, u2_np1, u3_np1  = ufl.split(u_np1)  #references to components of u_np1
u1_n, u2_n, u3_n        = ufl.split(u_n)    #these are ufl expressions. Not dolfinx functions. Cannot interpolate or do anything that requires a dolfinx function.
#ufl.split(u_n)[0] is symbolically same to u_n.split()[0] and u_n.sub(0).
#However, latter two are dolfinx function objects. u_n.split() uses u_n.sub(i) for all i.

# Interpolate initial condition, random values of Qxx Qxy which are smaller than 0.1
np.random.seed(8698)

def qinter(x):
    print("shape of Qxx field:", x.shape)
    ret = 0.2*np.random.rand(x.shape[1])-0.1
    print(ret[-1])
    return ret
def rhointer(x): # set density value to uniform seed value for now
    print("shape of density field:", x.shape)
    ret = rhoseed*np.ones(x.shape[1])
    print(ret[-1])
    return ret
    
u_n.sub(0).interpolate(qinter)
u_n.sub(1).interpolate(qinter)
u_n.sub(2).interpolate(rhointer)
u_n.x.scatter_forward()
u_np1.x.array[:] = u_n.x.array

# Define expressions used in variational forms

Pxx         = fem.Constant(mymesh, ScalarType(0.5)) # p is +x semiaxis
Pxy         = fem.Constant(mymesh, ScalarType(0))

a_np1       = (1 - u3_np1) # value of a_n+1 at t=0
TrQ_np1     = u1_np1**2 + u2_np1**2 # Qxx^2 +Qyy^2=S^2/2
rhoend_np1  = rhoisoend + ((rhonemend-rhoisoend) * TrQ_np1 * 4/(1+(2*TrQ_np1))) # value of rhoend_n+1 at t=0
Gamma_np1   = Gammas * ufl.tanh((rhoend_np1 - u3_np1)) # value of Gamma_n+1 at t=0

#Weak statement of the equations
F1  = inner(u1_np1, v1) * dx - inner(u1_n, v1) * dx + dt * Gamma_np1 * inner( (a_np1+ 2*TrQ_np1)*u1_np1,v1) * dx + dt * Gamma_np1 * K * inner(grad(u1_np1), grad(v1)) * dx - dt * Gamma_np1 * Pi * Pxx * v1 * dx
F2  = inner(u2_np1, v2) * dx - inner(u2_n, v2) * dx + dt * Gamma_np1 * inner( (a_np1+ 2*TrQ_np1)*u2_np1,v2) * dx + dt * Gamma_np1 * K * inner(grad(u2_np1), grad(v2)) * dx
F3  = inner(u3_np1, v3) * dx - inner(u3_n, v3) * dx - dt * inner(u3_np1, v3) * dx + dt* inner(u3_np1 * u3_np1 / rhoend_np1, v3) * dx
F   = F1 + F2 + F3

#Create nonlinear problem and Newton Solver
problem = NonlinearProblem(F, u_np1)
solver  = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-6
solver.convergence_criterion = "incremental"

# Output files
filexx = XDMFFile(MPI.COMM_WORLD, savedir + "Qxx_Output.xdmf", "w")
filexy = XDMFFile(MPI.COMM_WORLD, savedir + "Qxy_Output.xdmf", "w")
filerho = XDMFFile(MPI.COMM_WORLD, savedir + "rho_Output.xdmf", "w")

filexx.write_mesh(mymesh)
filexy.write_mesh(mymesh)
filerho.write_mesh(mymesh)

# Timestep begins
t   = 0.0
n   = 0

filexx.write_function(u_n.sub(0), t)
filexy.write_function(u_n.sub(1), t)
filerho.write_function(u_n.sub(2), t)

#setup a meshgrid
tol = 0.001

x   = np.linspace(0+tol, 1-tol, mx)
y   = np.linspace(0+tol, 1-tol, my)
xv, yv  = np.meshgrid(x,y)

points      = np.zeros((3, np.size(xv)))
points[0]   = xv.reshape(-1)
points[1]   = yv.reshape(-1)

from dolfinx import geometry
bb_tree    = geometry.BoundingBoxTree(mymesh, mymesh.topology.dim)
cells      = []
pointss    = []
cell_candidates = geometry.compute_collisions(bb_tree, points.T)
colliding_cells = geometry.compute_colliding_cells(mymesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i))>0:
            pointss.append(point)
            cells.append(colliding_cells.links(i)[0])
pointss = np.array(pointss, dtype=np.float64)

n_steps  = np.int32(n_steps)
Qxx_i   = np.zeros((n_steps+1, mx, my))
Qxy_i   = np.zeros((n_steps+1, mx, my))
rho     = np.zeros((n_steps+1, mx, my))

Qxx_i[0]  = u_n.sub(0).eval(pointss, cells).reshape(mx, my)
Qxy_i[0]  = u_n.sub(1).eval(pointss, cells).reshape(mx, my)
rho[0]    = u_n.sub(2).eval(pointss, cells).reshape(mx, my)

while (n < n_steps):
    n   += 1
    t   += dt
    r    = solver.solve(u_np1)
    if (n % 10 == 0):
        print(f'Step n {n}: num iterations: {r[0]}')
        
    u_n.x.array[:]  = u_np1.x.array     #copy over new soltn to current soltn for next loop
    
    filexx.write_function(u_n.sub(0), t)
    filexy.write_function(u_n.sub(1), t)
    filerho.write_function(u_n.sub(2), t)

    Qxx_i[n]  = u_n.sub(0).eval(pointss, cells).reshape(mx, my)
    Qxy_i[n]  = u_n.sub(1).eval(pointss, cells).reshape(mx, my)
    rho[n]    = u_n.sub(2).eval(pointss, cells).reshape(mx, my)

print(f'Solver finished')

filexx.close()
filexy.close()
filerho.close()

S_sqrd  = 2*(Qxx_i**2 + Qxy_i**2)
theta   = 0.5*np.arctan2(Qxy_i, Qxx_i)
nx      = np.cos(theta)
ny      = np.sin(theta)

rho.tofile(savedir+"rho.dat")
Qxx_i.tofile(savedir+"Qxx.dat")
Qxy_i.tofile(savedir+'Qxy.dat')

with open(savedir+'parameters.json', 'w') as f:
    json.dump(parameters, f)
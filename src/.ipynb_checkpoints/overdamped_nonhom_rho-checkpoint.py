import time
import ufl
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, VTXWriter, VTKFile
from ufl import grad, inner, dot, div
import numpy as np
import os, json

comm = MPI.COMM_WORLD

with open('paramsoverdamped.json') as jsonFile:
    parameters = json.load(jsonFile)

run       = parameters["run"]
T         = parameters["T"]        # final time
dt_dump   = parameters["dt_dump"]
n_steps   = parameters["n_steps"]  # number of time steps
K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
Gammas    = parameters["Gammas"]   # rate of Q alignment with mol field H
gamma     = parameters["gammaf"]   # traction coefficient
alpha     = parameters["alpha"]    # active contractile stress
chi       = parameters["chi"]      # coefficient of density gradients in Q's free energy
D         = parameters["D"]        # Density diffusion coefficient in density dynamics
lambd     = parameters["lambda"]   # flow alignment parameter
p0        = parameters["p0"]       # pressure when cells are close packed, should be very high
r_p       = parameters["r_p"]      # rate of pressure growth equal to rate of growth of cells
Pi        = parameters["Pi"]       # strength of alignment
rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
rhoseed   = parameters["rhoseed"] /rho_in     # seeding density, normalised by 100 mm^-2
rhoisoend = parameters["rhoisoend"] /rho_in   # jamming density
rhonemend = parameters["rhonemend"] /rho_in   # jamming density max for nematic substrate
mx        = np.int32(parameters["mx"])
my        = np.int32(parameters["my"])

dt        = T / n_steps     # time step size

savedir     = "../wo_pi/w_lambda/gammas_{:.2f}_rhoseed_{:.2f}_pi_{:.2f}/run_{}/".format(Gammas, rhoseed, Pi, run)

if comm.rank==0:
    os.makedirs(savedir)

#create mesh, define function spaces
mymesh  = mesh.create_unit_square(comm, mx, my, mesh.CellType.triangle, ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)
#mymesh.topology.create_connectivity_all()
dx      = ufl.Measure("dx", domain=mymesh)
ds      = ufl.Measure("ds", domain=mymesh)
normal  = ufl.FacetNormal(mymesh)
RE      = ufl.FiniteElement("Lagrange", mymesh.ufl_cell(), degree=2)    #scalar lagrange element for density
VE      = ufl.VectorElement("Lagrange", mymesh.ufl_cell(), degree=2)    #vector lagrange element for velocity


# Define test functions (no trials for nonlinear eqtns)
MF       = fem.FunctionSpace(mymesh, ufl.MixedElement(RE, VE, VE))
v_rho, v_Q, v_v = ufl.TestFunctions(MF)

# Define functions for solutions at previous and current time steps

u_np1   = fem.Function(MF)   #unknown Q, and density fields
u_n     = fem.Function(MF)   #solution from prev step

#unknown Q, rho, vel, and stress fields
(rho_np1, Q_np1, v_np1) = ufl.split(u_np1) #references to components of u_np1

#solutions from prev step
(rho_n, Q_n, v_n) = ufl.split(u_n) #references to components of u_n

#these references are ufl expressions. Not dolfinx functions. 
#Cannot interpolate or do anything that requires a dolfinx function.
#ufl.split(u_n)[0] is symbolically same to u_n.split()[0] and u_n.sub(0).
#However, latter two are dolfinx function objects. u_n.split() uses u_n.sub(i) for all i.

# Interpolate initial condition, random values of Qxx Qxy which are smaller than 0.1
#np.random.seed(898)

center_x=0.5; center_y=0.5; radius=0.9
def rhointer(x): # set density value to uniform seed value for now
    if comm.rank==0: print("shape of density field:", x.shape)
    distance = np.sqrt((x[0]-center_x)**2+(x[1]-center_y)**2)
    mean = rhoseed; std = rhoseed/2
    rhoseeds = np.abs(np.random.normal(mean, std, size=np.size(distance)))
    ret = np.where(distance <= radius, rhoseeds*(radius-distance)/radius, 0.001)
    return ret
def vinter(x):
    theta  = np.arctan2((x[1]-center_y), (x[0]-center_x))
    theta  = np.where(theta<0, theta+2*np.pi, theta)
    ret    = np.zeros([mymesh.geometry.dim, x.shape[1]])
    ret[0] = 0.01*np.cos(theta)
    ret[1] = 0.01*np.sin(theta)
    return ret
def qinter(x):
    #distance = np.sqrt((x[0]-center_x)**2+(x[1]-center_y)**2)
    theta    = np.pi*np.abs(np.random.normal(size=(x.shape[1])))
    ret      = 0.01*np.ones((mymesh.geometry.dim, x.shape[1]))
    ret[0]  *= np.cos(theta)
    ret[1]  *= np.sin(theta)
    return ret

u_n.sub(0).interpolate(rhointer)
u_n.sub(0).x.array[:] = abs(u_n.sub(0).x.array) #make sure all interpolated density values are positive, sometimes it messes it up when there are high gradients
u_n.sub(1).interpolate(qinter)
u_n.sub(2).interpolate(vinter)

u_n.x.scatter_forward()
u_np1.x.array[:] = u_n.x.array
u_np1.x.scatter_forward()



# Define expressions used in variational forms

Id          = ufl.Identity(mymesh.ufl_cell().geometric_dimension())
Pxx         = fem.Constant(mymesh, ScalarType(0.5)) # p is +x semiaxis
Pxy         = fem.Constant(mymesh, ScalarType(0))
Pij         = ufl.as_vector([Pxx, Pxy])


#a_np1       = (1 - rho_np1) # value of a_n+1 at t=0
#b_np1       = rho_np1 # value of b_np1 at t=0
S2_np1      = 2*inner(Q_np1, Q_np1)
S2_n        = 2*inner(Q_n, Q_n)
rhoend_np1  = rhoisoend + ((rhonemend-rhoisoend) * S2_np1 * 2/(1+(S2_np1))) # value of rhoend_n+1 at t=0, 2S^2/1+S^2
rhoend_n    = rhonemend#rhoisoend + ((rhonemend-rhoisoend) * S2_n * 2/(1+(S2_n)))
Gamma_np1   = Gammas * ufl.tanh((rhoend_np1 - rho_np1)) # value of Gamma_n+1 at t=0
Gamma_n     = Gammas * ufl.tanh((rhoend_n - rho_n)) # value of Gamma_n+1 at t=0

divv_np1    = div(v_np1)
gradv_np1   = grad(v_np1)
#note: grad(Tensor) is implemented in ufl as: del_k T_ij = gradT_ijk
#ctd: grad(Vector) is implemented in ufl as: del_k V_i = gradV_ik
#ctd: div(Tensor) is implemented in ufl as: del_jT_ij (which is the correct form)
#ctd: so v.grad(v)=v_k del_k v_j should be written as dot(grad(v), v) and NOT dot(v, grad(v))
#ctd: and v.grad(T)=v_k del_k T_ij should be implemented as dot(grad(T), v)
#ctd: this will be important when writing weak forms.

kappas_np1  = 0.5 * (gradv_np1 + gradv_np1.T - divv_np1*Id)
#kappaa_np1  = 0.5 * (gradv_np1 - gradv_np1.T)

kappas_n = ufl.as_vector([v_n[0].dx(0)-v_n[1].dx(1), v_n[0].dx(1)+v_n[1].dx(0)])/2

QM_n        = ufl.as_matrix([[Q_n[0], Q_n[1]], [Q_n[1], -Q_n[0]]])

p_np1       = p0*ufl.exp(r_p*(rho_np1-rhoend_np1)) #for now rho_c is rho_end as I figure this out
p_n         = p0*ufl.exp(r_p*(rho_n-rhoend_n)) 

n_rho_vector = ufl.as_vector([normal[0]*rho_n.dx(0)-normal[1]*rho_n.dx(1) , normal[0]*rho_n.dx(1)+normal[1]*rho_n.dx(0)])
#Weak statement of the equations
#weak statement for Q
F1  = inner(Q_np1, v_Q) * dx - inner(Q_n, v_Q) * dx 
F1 += dt * inner(dot(grad(Q_n), v_n), v_Q) * dx - dt*2*inner(Qrot_n, v_Q)*dx 
F1 -= dt*inner(lambd*kappas_n, v_Q)*dx 
F1 += dt*inner(Gamma_n*(1-rho_n+(rho_n*S2_n))*Q_np1, v_Q)*dx
F1 += dt*inner(Gamma_n*K*grad(Q_np1), grad(v_Q))*dx #- dt*inner(Gamma_n*Pi*Pij, v_Q)*dx
F1 += dt*0.5*chi*Gamma_n*inner(n_rho_vector, v_Q)*ds
F1 += dt*0.5*chi*Gamma_n*(-rho_n.dx(0)*v_Q[0].dx(0)+rho_n.dx(1)*v_Q[0].dx(1)-rho_n.dx(1)*v_Q[1].dx(0)-rho_n.dx(0)*v_Q[1].dx(1))*dx

#weak statement for v
F2  = gamma*inner(rho_n*v_np1, v_v)*dx
F2 += r_p*inner(p_n*grad(rho_n), v_v)*dx
F2 -= alpha*inner(div(QM_n), v_v)*dx

#weak statement for rho
F3  = inner(rho_np1, v_rho)* dx - inner(rho_n, v_rho)* dx
F3 += dt* (alpha/gamma)* inner(dot(normal, div(QM_n)), v_rho)*ds - dt* (alpha/gamma) *inner(div(QM_n), grad(v_rho))*dx
F3 += dt* (r_p/gamma) *inner(p_n*grad(rho_np1), grad(v_rho))* dx
F3 += dt* D*inner(grad(rho_np1), grad(v_rho))*dx
F3 += - dt * inner(rho_np1, v_rho)* dx + dt* inner(rho_np1 * rho_np1 / rhoend_n, v_rho)* dx

F   = F1 + F2 + F3
print('all forms created')
#Create nonlinear problem and Newton Solver
problem = NonlinearProblem(F, u_np1)
solver  = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-5
solver.convergence_criterion = "incremental"
print('solver created')

def savefile(filename, function, n):
    #file = XDMFFile(MPI.COMM_WORLD, savedir + filename + "_Output_{:d}.xdmf".format(n), "w")
    #file.write_mesh(mymesh)
    #file.write_function(function, n)
    
    #file = VTXWriter(mymesh.comm, savedir + filename + "{:d}.bp".format(n), [function])
    #file.write(n)
    
    file = VTKFile(mymesh.comm, savedir + filename + "_{:d}.pvd".format(n), "w")
    file.write_mesh(mymesh)
    file.write_function(function, n)
    file.close()
    
#setup a meshgrid
tol = 0.001

x   = np.linspace(0+tol, 1-tol, mx)
y   = np.linspace(0+tol, 1-tol, my)
xv, yv  = np.meshgrid(x,y)

points      = np.zeros((3, np.size(xv)))
points[0]   = xv.reshape(-1)
points[1]   = yv.reshape(-1)

from dolfinx import geometry
bb_tree    = geometry.bb_tree(mymesh, mymesh.topology.dim)
cells      = []
pointss    = []
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
colliding_cells = geometry.compute_colliding_cells(mymesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i))>0:
            pointss.append(point)
            cells.append(colliding_cells.links(i)[0])
pointss = np.array(pointss, dtype=np.float64)

def savedatscalar(filename, function, n):
    field = function.eval(pointss, cells).reshape(mx, my)
    np.savetxt(savedir+filename+'_{:d}.csv'.format(n), field)
def savedatvector(filename, function, n):
    field = function.eval(pointss, cells).reshape(mx, my, 2)
    np.savetxt(savedir+filename+'x_{:d}.csv'.format(n), field[:,:,0])
    np.savetxt(savedir+filename+'y_{:d}.csv'.format(n), field[:,:,1])
    
# Timestep begins
t       = 0.0
n       = 0
ndump   = 0
dn_dump = int(dt_dump/dt)

with open(savedir+'parameters.json', 'w') as f:
    json.dump(parameters, f)
print('parameter dump created')
savefile('rho', u_n.sub(0).collapse(), ndump); savedatscalar('rho', u_n.sub(0), ndump)
savefile('v', u_n.sub(2).collapse(), ndump); savedatvector('v', u_n.sub(2), ndump)
savefile('Q', u_n.sub(1).collapse(), ndump); savedatvector('Q', u_n.sub(1), ndump)

while (n < n_steps):
    n   += 1
    t   += dt
    #T1 = time.time()
    r    = solver.solve(u_np1)
    #T2 = time.time()
    #print('time:' , t, 'time taken:', T2-T1)
    #u_n.sub(0).collapse().x.array[:] = np.abs(u_n.sub(0).collapse().x.array)   #make sure density is positive, no negative fluctuations
    u_n.x.array[:]  = u_np1.x.array     #copy over new soltn to current soltn for next loop
    u_n.x.scatter_forward()

    if ( n % dn_dump == 0):
        ndump+=1
        if comm.rank==0:
            print(f'Time t {t}; Step n {n}; Num iterations {r[0]}')
        savefile('rho', u_n.sub(0).collapse(), ndump) ; savedatscalar('rho', u_n.sub(0), ndump)
        savefile('v', u_n.sub(2).collapse(), ndump) ; savedatvector('v', u_n.sub(2), ndump)
        savefile('Q', u_n.sub(1).collapse(), ndump) ; savedatvector('Q', u_n.sub(1), ndump)
    
print(f'Solver finished')

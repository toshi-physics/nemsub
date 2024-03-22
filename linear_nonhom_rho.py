import ufl
import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from ufl import grad, inner, dot, div
import numpy as np
import os, json

comm = MPI.COMM_WORLD

with open('paramsnew.json') as jsonFile:
    parameters = json.load(jsonFile)

run       = parameters["run"]
T         = parameters["T"]        # final time
dt_dump   = parameters["dt_dump"]
n_steps   = parameters["n_steps"]  # number of time steps
K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
Gammas    = parameters["Gammas"]   # rate of Q alignment with mol field H
gamma     = parameters["gammaf"]   # traction coefficient
lambd     = parameters["lambda"]
mu        = parameters["mu"]
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

savedir     = "full/gammas_{:.2f}_rhoseed_{:.2f}_pi_{:.2f}/run_{}/".format(Gammas, rhoseed, Pi, run)

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

center_x=0.5; center_y=0.5; radius=0.2
def rhointer(x): # set density value to uniform seed value for now
    if comm.rank==0: print("shape of density field:", x.shape)
    distance = np.sqrt((x[0]-center_x)**2+(x[1]-center_y)**2)
    rhoseeds = rhoseed*np.abs(np.random.normal(size=np.size(distance)))
    ret = np.where(distance <= radius, rhoseeds, 0.01)
    return ret
def vinter(x):
    distance = np.sqrt((x[0]-center_x)**2+(x[1]-center_y)**2)
    ret = 0.01*np.random.normal(size=(mymesh.geometry.dim, x.shape[1]))
    ret[0] = np.where(distance<=radius, ret[0], 0)
    ret[1] = np.where(distance<=radius, ret[1], 0)
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

H_np1       = -(1-rho_np1+rho_np1*S2_np1)*Q_np1  + Pi*Pij
H_n         = -(1-rho_n+rho_n*S2_n)*Q_n + Pi*Pij
HM_np1      = ufl.as_matrix([[H_np1[0], H_np1[1]],[H_np1[1], -H_np1[0]]])
HM_n        = ufl.as_matrix([[H_n[0], H_n[1]],[H_n[1], -H_n[0]]])
Qrot_np1    = 0.5*(v_n[1].dx(0)-v_n[0].dx(1)) * ufl.as_vector([Q_np1[1], -Q_np1[0]]) #cross product in ufl is only for 3D vectors
Qrot_n      = 0.5*(v_n[1].dx(0)-v_n[0].dx(1)) * ufl.as_vector([Q_n[1], -Q_n[0]]) #cross product in ufl is only for 3D vectors

p_np1       = p0*ufl.exp(r_p*(rho_np1-rhoend_np1)) #for now rho_c is rho_end as I figure this out
p_n         = p0*ufl.exp(r_p*(rho_n-rhoend_n)) 

#sigma_np1   = mu*kappas_np1 - ufl.sqrt(S2_np1)*lambd*H_np1#-p_np1*Id
#divsigma_np1= mu*0.5*div(grad(v_np1))-lambd*dot(HM_np1, grad(ufl.sqrt(S2_np1)))-lambd*ufl.sqrt(S2_np1)*ufl.div(HM_np1)-r_p*p_np1*grad(rho_np1)

#Weak statement of the equations
#weak statement for Q
F1  = inner(Q_np1, v_Q) * dx - inner(Q_n, v_Q) * dx 
F1 += dt * inner(dot(grad(Q_n), v_n), v_Q) * dx - dt*2*inner(Qrot_n, v_Q)*dx 
#F1 -= dt*inner(ufl.sqrt(S2_n)*lambd*kappas_n, v_Q)*dx 
F1 += dt*inner(Gamma_n*(1-rho_n+(rho_n*S2_n))*Q_np1, v_Q)*dx
F1 += dt*inner(Gamma_n*K*grad(Q_np1), grad(v_Q))*dx #- dt*inner(Gamma_n*Pij, v_Q)*dx

#weak statement for v
F2  = inner(rho_n*v_np1, v_v)*dx -inner(rho_n*v_n, v_v)*dx + dt*inner(rho_n*ufl.dot(grad(v_np1), v_np1) , v_v)*dx 
F2 -= dt* inner(v_np1*rho_n, v_v)* dx 
F2 += dt* inner(v_np1*rho_n * rho_n / rhoend_n, v_v)* dx
F2 += dt* 0.5*mu* inner(grad(v_np1), grad(v_v))*dx
F2 += dt* 0.5*mu* inner(div(v_np1), div(v_v))*dx - dt* 0.5*mu* inner(div(v_np1), dot(normal, v_v))*ds
F2 += dt*r_p*inner(p_n*grad(rho_n), v_v)*dx
F2 += dt*gamma*inner(rho_n*v_np1, v_v)*dx
#+ dt*np.sqrt(2)*lambd*inner(dot(dot(Q_n, grad(Q_n))/ufl.sqrt(S2_n), HM_n), v_v)*dx
#+ dt*lambd*inner(ufl.sqrt(S2_n)*div(HM_n), v_v)*dx

#weak statement for rho
F3  = inner(rho_np1, v_rho)* dx - inner(rho_n, v_rho)* dx + dt* inner(dot(grad(rho_np1), v_n), v_rho)* dx 
F3 += dt* inner(div(v_n)*rho_np1, v_rho)* dx - dt * inner(rho_np1, v_rho)* dx 
F3 += dt* inner(rho_np1 * rho_np1 / rhoend_n, v_rho)* dx

F   = F1 + F2 + F3

#Create nonlinear problem and Newton Solver
problem = NonlinearProblem(F, u_np1)
solver  = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-5
solver.convergence_criterion = "incremental"

def savefile(filename, function, n):
    file = XDMFFile(MPI.COMM_WORLD, savedir + filename + "_Output_{:d}.xdmf", "w")
    file.write_mesh(mymesh)
    file.write_function(function, n)
    file.close()

# Timestep begins
t       = 0.0
n       = 0
ndump   = 0
dn_dump = int(dt_dump/dt)

with open(savedir+'parameters.json', 'w') as f:
    json.dump(parameters, f)
savefile('rho', u_n.sub(0), ndump)
savefile('v', u_n.sub(2), ndump)
savefile('Q', u_n.sub(1), ndump)

while (n < n_steps):
    n   += 1
    t   += dt
    r    = solver.solve(u_np1)

    #u_n.sub(0).collapse().x.array[:] = np.abs(u_n.sub(0).collapse().x.array)   #make sure density is positive, no negative fluctuations
    u_n.x.array[:]  = u_np1.x.array     #copy over new soltn to current soltn for next loop
    u_n.x.scatter_forward()

    if ( n % dn_dump == 0):
        ndump+=1
        if comm.rank==0:
            print(f'Time t {t}; Step n {ndump}; Num iterations {r[0]}')
        savefile('rho', u_n.sub(0), ndump)
        savefile('v', u_n.sub(2), ndump)
        savefile('Q', u_n.sub(1), ndump)
    
print(f'Solver finished')
"""
Dam Break by yiming liu

2021/5/28
"""

from dolfin import *
from dolfin.cpp.mesh import cells
from mshr import *
from ufl import indices
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
#from math import exp
import time

class CharacterLevelSetFunction(UserExpression):
	'''
	initialize the level set function
	Firstly, determine the position of an arbitrary point in the domain
	Secondly, calculate the distance to the minimum distance to the interface
	Finally, calculate the CharacterLevelSetFunction, 
	in the interface =1, out the interface = 0, on the transition area of interface it is continuous
	'''
	def __init__(self, param1, param2, param3, **kwargs):
		super().__init__(**kwargs)
		self.param1 = param1 # x boundary of interface
		self.param2 = param2 # y boundary of interface
		self.param3 = param3 # epsilon: thickness parameter

	def eval(self, values, x):
		if x[0] <= self.param1 and x[1] <= self.param2:
			phi_d = - min(self.param1 - x[0], self.param2 - x[1])
			values[0] = 1/(1+exp(phi_d/self.param3))
		if x[0] > self.param1 and x[1] > self.param2:
			dx=x[0]-self.param1
			dy=x[1]-self.param2
			phi_d = pow(pow(dx,2)+pow(dy,2),0.5)
			values[0] = 1/(1+exp(phi_d/self.param3))
		if x[0] > self.param1 and x[1] <= self.param2:
			phi_d = x[0] - self.param1
			values[0] = 1/(1+exp(phi_d/self.param3))
		if x[0] <= self.param1 and x[1] > self.param2:
			phi_d = x[1] - self.param2
			values[0] = 1/(1+exp(phi_d/self.param3))

processID =  MPI.comm_world.Get_rank()
set_log_level(40) #50, 40, 30, 20, 16, 13, 10
t_end = 5.#8
dt = 1./200. 
t=0.
foldername = "data/Dambreak"
nn=0

res=50 #resolution
scale=1
mesh = RectangleMesh(Point(0,0), Point(scale,scale),res,res,"crossed")
plot(mesh)
n = FacetNormal(mesh)

i, j, k, l = indices(4)
delta = Identity(2)

scalar = FiniteElement('P', mesh.ufl_cell(), 1) #pressure level set function
vector = VectorElement('P', mesh.ufl_cell(), 2) #velocity
vector1 = VectorElement('P', mesh.ufl_cell(), 1) #interface normal
mixed_element = MixedElement([scalar, vector, vector])
ScalarSpace = FunctionSpace(mesh, scalar) #pressure
ScalarSpace1 = FunctionSpace(mesh, scalar) #level set function
VectorSpace = FunctionSpace(mesh, vector) #velocity
VectorSpace1 = FunctionSpace(mesh, vector1) #interface normal

## Define boundary condition

wallsx = 'on_boundary && (near(x[0], 1, 1e-14) || near(x[0], 0, 1e-14) )' #x=1 and x=0
wallsy = 'on_boundary && (near(x[1], 1, 1e-14) || near(x[1], 0, 1e-14) )' #y=1 and y=0
bc = []
bc2 = []
# free slip boundary condition dot(u,n)=0
bc2.append(DirichletBC(VectorSpace.sub(0), Constant(0), wallsx)) 
bc2.append(DirichletBC(VectorSpace.sub(1), Constant(0), wallsy))


v = TestFunction(VectorSpace)
q = TestFunction(ScalarSpace)
u_t = Function(VectorSpace) #tentative velocity
u_c = Function(VectorSpace) #tentative velocity after correction
u_c0 = Function(VectorSpace) #tentative velocity after correction
p = Function(ScalarSpace)
p0 = Function(ScalarSpace)

del_phi_c = TestFunction(ScalarSpace1) #level set function
phi_c = Function(ScalarSpace1) #level set function
phi_cr = Function(ScalarSpace1) #level set function after reinitialization
phi_c0 = Function(ScalarSpace1)
phi_cr0 = Function(ScalarSpace1) #level set function after reinitialization

del_n_phi = TestFunction(VectorSpace1) 
n_phi = Function(VectorSpace1) #normal of interface calculate from level set function

V = FunctionSpace(mesh, 'P', 1)


## Define initial conditions
epsilon=0.5*pow(scale/res,0.9)#related to the size of element
interface_x=0.3*scale
interface_y=0.5*scale
initial_phi_c = CharacterLevelSetFunction(param1=interface_x, param2=interface_y, param3=epsilon, element=V.ufl_element(),degree=1)#initial level set function
initial_v = Expression(('0.', '0.'), degree=1)
initial_p = Expression('(x[0] < x0 && x[1] < y0) ? (y0-x[1])*1.0 : 0.0',x0=interface_x,y0=interface_y, degree=1) #initial the pressure field of water 

#%%
## initialization

# project(initial_phi_c, ScalarSpace1)
# project(initial_v, VectorSpace)
# project(initial_p, ScalarSpace)
phi_c.assign(project(initial_phi_c, ScalarSpace1))
phi_c0.assign(project(initial_phi_c, ScalarSpace1))
u_c0.assign(project(initial_v, VectorSpace))
p0.assign(project(initial_p*initial_phi_c, ScalarSpace)) 
#mutiply the inital level-set value can make the initial calculation step more stable, but it has no effect on the result
# can also use p0.assign(project(initial_p, ScalarSpace)) 


# set parameters
Re=2000.
Fr=0.7
rho0=1.
nu0=1.
rho1=0.001
nu1=1.
# write file
file_pp = File(foldername+'/p_Re%d' %Re + '_Fr%.3s.pvd'%Fr)
file_vv = File(foldername+'/v_Re%d' %Re + '_Fr%.3s.pvd'%Fr)
file_c = File(foldername+'/c_Re%d' %Re + '_Fr%.3s.pvd'%Fr)
file_c << (phi_c0,t)
file_pp << (p0,t)

file_p = XDMFFile(foldername+'/p_Re%d' %Re + '_Fr%.3s.xdmf'%Fr)
file_v = XDMFFile(foldername+'/v_Re%d' %Re + '_Fr%.3s.xdmf'%Fr)
file_phi = XDMFFile(foldername+'/phi_Re%d' %Re + '_Fr%.3s.xdmf'%Fr)
#file_p.write_checkpoint(p, "p", t, append=False)
#file_v.write_checkpoint(u_c, "v", t, append=False)
#file_phi.write_checkpoint(phi_c, "phi", t, append=False)


g_=-1.
rou= phi_c*rho0+(1-phi_c)*rho1
mu= phi_c*nu0+(1-phi_c)*nu1

ndim = mesh.geometry().dim()

# viscous operator
def a(phi,chi, psi): return Constant(0.5) * inner(grad(phi) + grad(chi).T, grad(psi) + grad(psi).T)
# divergence operator
def b(phi, psi): return inner(div(phi), psi)
# non-linear convection operator
def c(phi, chi, psi): return dot(dot(grad(chi), phi), psi)
# Crank-Nicholson schema average
def d(phi, psi): return Constant(0.5) * (phi + psi)

#n_phi=grad(phi_c)/pow(dot( grad(phi_c), grad(phi_c) ), 0.5)
g = Expression(('0.','g'),degree=1,g=g_)

# interface normal n = ▽ϕ/|ϕ|
F_n_phi = (dot(grad(phi_c)/pow(dot( grad(phi_c), grad(phi_c) ), 0.5), del_n_phi) - dot(n_phi,del_n_phi)) * dx

## Projection method
F_momentum = (rou*dot((u_t-u_c0),v)/dt + rou*c(u_t, u_t, v) + (1. / Re) * mu*a(u_t,u_t, v) -  b(v, p0) - (1./(Fr*Fr))*rou*(dot(g,v))) * dx
#F_momentum = (rou*dot((u_t-u_c0),v)/dt - dot(u_t, div(rou*outer(v,u_t)) ) + (1. / Re) * mu*a(u_t,u_t, v) -  b(v, p0) - (1./(Fr*Fr))*rou*(dot(g,v))) * dx
# pressure correction
F_pressure = (inner(grad(p-p0),grad(q))/rou + (1/dt)*div(u_t)*q) * dx
#velocity correction
F_velocity = ((1/dt)*inner(u_c-u_t,v) + inner(grad(p-p0),v)/rou) * dx

## advection and reinitialization Crank-Nicholson schema
F_advection = ( (1/dt)*inner(phi_c-phi_c0, del_phi_c) - 0.5*inner( (phi_c+phi_c0), inner( u_c, grad(del_phi_c) ) ) ) * dx

dts=0.5*pow(scale/res,1.1) #sub-time step related to the mesh resolution
F_reinitialization = ( (1/dts)*inner(phi_cr - phi_cr0, del_phi_c) - inner(d(phi_cr, phi_cr0)*(1-d(phi_cr, phi_cr0)), dot(n_phi, grad(del_phi_c))) + epsilon*inner( grad(d(phi_cr, phi_cr0)), grad(del_phi_c))  ) * dx

#%%
while t < t_end:
	t += dt
	nn+= 1
	begin("Computing interface normal")
	solve(F_n_phi==0, n_phi)
	
	begin("Computing tentative velocity")
	solve(F_momentum==0, u_t, bc2)
	
	begin("Computing pressure correction")
	solve(F_pressure==0, p, bc)
	
	begin("Computing velocity correction")
	solve(F_velocity==0, u_c, bc2)
		
	begin("Computing Level Set Fucntion")
	solve(F_advection==0, phi_c, bc)
	
	nreinitial=0
	maxstep=1000
	tol=6*epsilon
	phi_cr0.assign(phi_c)
	
	begin("Reinitialization")
	while nreinitial < maxstep:
		nreinitial += 1
		solve(F_reinitialization==0, phi_cr, bc)
		#error=project(abs(phi_cr-phi_cr0),ScalarSpace)
		error=(phi_cr.vector() - phi_cr0.vector()).max() / dts
		if error < tol: break
		if processID == 0:
			if nreinitial==1: print('Error at substep 1: %7f' % error)
			
		phi_cr0.assign(phi_cr)
		
	if processID == 0: print('Error in the end: %7f' % error)
	if processID == 0: print('Number of iterations for reinitialization: %d' % nreinitial)
	dofs=len(u_t.vector())
	if processID == 0: print('time: %f s,  with %d dofs' % (t,dofs) )
	
	phi_c.assign(phi_cr)
	u_c0.assign(u_c)
	p0.assign(p)
	phi_c0.assign(phi_c)

	if nn == 1 or (nn % 20 == 0 and nn<=10000):
		#file_p.write_checkpoint(p, "p", t, append=True)
		#file_v.write_checkpoint(u_c, "v", t, append=True)
		#file_phi.write_checkpoint(phi_c, "phi", t, append=True)
		file_pp << (p0,t)
		file_c << (phi_c0,t)
		file_vv << (u_c,t)

file_pp << (p0,t)
file_c << (phi_c0,t)
file_vv << (u_c,t)

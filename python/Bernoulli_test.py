"""
Bernoulli test by yiming liu

I write these according to Emek's codes partially

supply code
"""

from dolfin import *
import mshr
from ufl import indices
import matplotlib.pyplot as plt
%matplotlib inline
import numpy
import time
'''
v2
1 1 "out"
1 2 "walls"
1 3 "bottom"
1 4 "in"
2 1 "water"

'''

processID =  MPI.comm_world.Get_rank()
set_log_level(50) #50, 40, 30, 20, 16, 13, 10

foldername = "free_jet/"
t_end = 1.#8
dt = 1./100. #1600
t=0.

#30mm*30mm, outlet deepth:1mm
fname = "geo/free_jet_v2_2W"
mesh = Mesh(fname + '.xml')
cells = MeshFunction('size_t', mesh , fname + '_physical_region.xml')
facets = MeshFunction('size_t', mesh , fname + '_facet_region.xml')
n = FacetNormal(mesh)

i, j, k, l = indices(4)
delta = Identity(2)

scalar = FiniteElement('P', triangle, 1)
vector = VectorElement('P', triangle, 2)
#equal order elements are known to be unstable for an incompressible fluid, quadratic element for the velocity and linear for the pressure
mixed_element = MixedElement([scalar, vector])
Space = FunctionSpace(mesh, mixed_element)


da = Measure( 'ds' , domain=mesh , subdomain_data=facets )
dv = Measure( 'dx' , domain=mesh , subdomain_data=cells )

bottom = CompiledSubDomain('near(x[0],50) &&on_boundary')

# Define inflow profile
rho = 1.
mu = 0.01 

la = 0
v_in = Expression(('v','0.'), degree=1,v=0)
bc = []

# Define boundary conditions
bc.append( DirichletBC(Space.sub(1), v_in,facets, 4) )
bc.append( DirichletBC(Space.sub(1), Constant((0.,0.)), facets, 3) )
bc.append( DirichletBC(Space.sub(1), Constant((0.,0.)), facets, 2) )
bc.append( DirichletBC(Space.sub(0), 0., facets,1) )
bc.append( DirichletBC(Space.sub(0), 0., facets,4) )

dunkn = TrialFunction(Space)
test = TestFunction(Space)
unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)


file_p = File(foldername+'p.pvd')
file_v = File(foldername+'v.pvd')


p0, v0 = split(unkn0)
p, v = split(unkn)
del_p, del_v = split(test)



#units mass:g, s, mm, Pa,  Gravity:g= mm/s^2

d = as_tensor( 1./2.*(v[i].dx(j)+v[j].dx(i)) , [i,j] )
tau = as_tensor( 2.*mu*d[i,j] , [i,j] )
#tau = as_tensor( mu*v[i].dx(j) , [i,j] )
g = Expression(('980*time','0.'),degree=1,time=0)
F_1 = v[i].dx(i)*del_p*dv
#F_2 = ( ( rho*(v-v0)[j]/dt -rho*g[j] + rho*v[i]*(v[j].dx(i)) + p.dx(j) ) *del_v[j]  + tau[i,j]*del_v[j].dx(i) )*dv 
F_2 = ( ( rho*(v-v0)[j]/dt -rho*g[j] + rho*v[i]*(v[j].dx(i)) + p.dx(j) ) *del_v[j]  + tau[i,j]*1./2.*( del_v[j].dx(i) + del_v[i].dx(j) )*dv 
#because the antisymmetric part cancels, tau[i, j], with any other second rank tensor, 
#like' del_v[j].dx(i) ', only the symmetric part of this kind of tensor, like 1./2.*( del_v[j].dx(i) + del_v[i].dx(j) ), matters.
Form = F_1 + F_2 

Gain = derivative(Form, unkn, dunkn)
while t < t_end:
	t += dt
	g.time=t #it cannot be sovled directly, maybe because the value of gravity is large 

	solve(Form==0, unkn, bc, J=Gain, solver_parameters={"newton_solver":{"linear_solver": "lu", "relative_tolerance": 1e-5}},\
	      form_compiler_parameters={"cpp_optimize": True,"representation": "uflacs"}) #uflacs
	dofs=len(unkn.vector())

	if processID == 0: print('time: %f s,  with %d dofs' % (t,dofs) )
	v1=assemble(dot(v,n)*da(1))#flux of outlet
	

	p_out, v_out = unkn.split(deepcopy=True)
	#print(v_out(Point(50.,4.9))[0])
	v_in.v=v1/30#velocity of inlet
	#file_p.write_checkpoint(p_out, "p", t, append=True)
	#file_v.write_checkpoint(v_out, "v", t, append=True)
	unkn0.assign(unkn)

print(v_out(Point(31,15)))
print(p_out(Point(30,1)))
file_p << (p_out,t)
file_v << (v_out,t)

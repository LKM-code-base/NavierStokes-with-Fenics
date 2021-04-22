"""
Crusher code by yiming liu

This work is according to Mariia Fomicheva's theory in paper 'Micropolar medium in a funnel-shaped crusher' published in 2021

"""

from dolfin import *
import mshr
from ufl import indices
import matplotlib.pyplot as plt
%matplotlib inline
import numpy
import time
'''
1 1 "walls"
1 2 "inlet"
1 3 "outlet"
2 1 "matter"

'''
#solver parameters settings according to Emek's work in 2019 'An Accurate Finite Element Method for the Numerical Solution 
# of Isothermal and Incompressible Flow of Viscous Fluid'
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["mesh_partitioner"] = "SCOTCH"#"ParMETIS"
parameters["ghost_mode"] = 'shared_vertex'

krylov_params = parameters["krylov_solver"]
krylov_params["relative_tolerance"] = 1E-3
krylov_params["absolute_tolerance"] = 1E-6
krylov_params["monitor_convergence"] = False
krylov_params["maximum_iterations"] = 15000

processID =  MPI.comm_world.Get_rank()
set_log_level(50) #50, 40, 30, 20, 16, 13, 10

foldername = "crusher_2w/"
t_end = 1.05#8
dt = 1./1000. #1600
t=0.
n=0

fname = "geo/crusher_2w"
mesh = Mesh(fname + '.xml')
cells = MeshFunction('size_t', mesh , fname + '_physical_region.xml')
facets = MeshFunction('size_t', mesh , fname + '_facet_region.xml')
#nn = FacetNormal(mesh)
#on the server benz, the xml cannot be read, How to deal with this problem?

i, j, k, l = indices(4)
delta = Identity(2)

scalar = FiniteElement('P', triangle, 1)
vector = VectorElement('P', triangle, 1)
mixed_element = MixedElement([scalar, scalar, vector])#p,J,v
Space = FunctionSpace(mesh, mixed_element)

da = Measure( 'ds' , domain=mesh , subdomain_data=facets )
dv = Measure( 'dx' , domain=mesh , subdomain_data=cells )

# Define the values of dimensionless quantities
g_=0.907
a_=1.575
J_up=20.
J1=1.
P_up=0.
L=0.5
H=1.
lambd=0.125
mu0=0.01
mu1=0.1

bc = []
# Define boundary conditions
bc.append( DirichletBC(Space.sub(1), J_up,facets, 2) )
bc.append( DirichletBC(Space.sub(2), Constant((0.,0.)), facets, 1) )
bc.append( DirichletBC(Space.sub(0), P_up, facets,2) )
bc.append( DirichletBC(Space.sub(0), 0., facets,3) )

dunkn = TrialFunction(Space)
test = TestFunction(Space)
unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

'''
file_p = XDMFFile(foldername+'/p.xdmf')
file_J = XDMFFile(foldername+'/J.xdmf')
file_v = XDMFFile(foldername+'/v.xdmf')
p_out, J_out, v_out = unkn.split(deepcopy=True)
file_p.write_checkpoint(p_out, "p", t, append=False)
file_J.write_checkpoint(J_out, "J", t, append=False)
file_v.write_checkpoint(v_out, "v", t, append=False)
'''
file_p = File(foldername+'p.pvd')
file_J = File(foldername+'J.pvd')
file_v = File(foldername+'v.pvd')
'''
The xdmf is better, becasue the result only have two files, .pvd stores the results of each step separately. 
But the xdmf result sovled by my computer cannot open by paraview 
(I try to open it by paraview5.6 and 5.9 under MacOS and Ubuntu, 
 none of these work. maybe because of the version of the annaconda, My fenics is built under anaconda)
The xdmf result files from server benz is ok, but here, I use the .xml 
and when I run this code on benz, it will report an error

'''

p0, J0, v0 = split(unkn0)
p, J, v = split(unkn)
del_p, del_J, del_v = split(test)


#dimensionless quantities

mu = mu0 + (mu1 - mu0)*exp(-lambd*(J-J1)/J1)
d = as_tensor( 1./2.*(v[i].dx(j)+v[j].dx(i)) , [i,j] )
tau = as_tensor( 2.*mu*d[i,j] , [i,j] )
#tau = as_tensor( mu*v[i].dx(j) , [i,j] )
g = Expression(('0.','-g'),degree=1,g=0.907)
F_1 = v[i].dx(i)*del_p*dv
F_2 = ( ( (v-v0)[j]/dt -g[j] + (v[i]*v[j]).dx(i) + p.dx(j) )*del_v[j]  + tau[i,j]*del_v[j].dx(i) )*dv 
F_3 = ( (J-J0)/dt + v[i]*J.dx(i) + a_*p*(J-1) )*del_J*dv
Form = F_1 + F_2 + F_3

Gain = derivative(Form, unkn, dunkn)
while t < t_end:
	t += dt
	n += 1
	solve(Form == 0, unkn, bcs=bc, J=Gain, solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1E-3,\
			"krylov_solver": krylov_params, "maximum_iterations":30, "error_on_nonconvergence": True} } )

	#solve(Form==0, unkn, bc, J=Gain, solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3}},\
	#      form_compiler_parameters={"cpp_optimize": True,"representation": "uflacs"}) #uflacs
	dofs=len(unkn.vector())

	if processID == 0: print('time: %f s,  with %d dofs' % (t,dofs) )
	
	#I don't want to store every step's result
	if n % 50 == 0 and n<=1.05:
		p_out, J_out, v_out = unkn.split(deepcopy=True)
		'''
		file_p.write_checkpoint(p_out, "p", t, append=True)
		file_J.write_checkpoint(J_out, "J", t, append=True)
		file_v.write_checkpoint(v_out, "v", t, append=True)
		'''
		file_p << (p_out,t)
		file_J << (J_out,t)
		file_v << (v_out,t)

	unkn0.assign(unkn)
'''
In paper, it study the result from four time points:0.00105, 0.0525, 0.105, 105
the time interval from 0.105 to 105 is long, The intermediate result is not needed, save the final result directly 
'''
#I didnt run it to the time point 105 becasue it takes a lot of time. After I solve the problem about xml, I can try this on the server benz.
'''
file_p << (p_out,t)
file_J << (J_out,t)
file_v << (v_out,t)
'''

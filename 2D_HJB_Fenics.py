
"""
Solving 2D HJB PDE
"""

from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
# Create mesh and build function space
nx, ny = 149,149
mesh = RectangleMesh(Point(-3., -3.), Point(3., 3.), nx, ny) # 
# keep dof in serial order (see https://fenicsproject.org/qa/3258/manually-setting-values-to-a-function/)
parameters["reorder_dofs_serial"] = False 
V = FunctionSpace(mesh, "Lagrange", 1)

# Create boundary markers
tol = 1E-14

tdim = mesh.topology().dim()
boundary_parts = MeshFunction('size_t', mesh, tdim-1)
left   = AutoSubDomain(lambda x: near(x[0], 0.0, tol))
right  = AutoSubDomain(lambda x: near(x[0], 1.0, tol))
bottom = AutoSubDomain(lambda x: near(x[1], 0.0, tol))
top    = AutoSubDomain(lambda x: near(x[1], 1.0, tol))

left.mark(boundary_parts, 1)
right.mark(boundary_parts, 1)
bottom.mark(boundary_parts, 1)
bottom.mark(boundary_parts, 1)

boundary_file = File("conc_diff/boundaries.pvd")
boundary_file << boundary_parts
# Terminal condition and right-hand side
spc = SpatialCoordinate(mesh)
denom = (2*pi)* 0.4
num   = 50* exp(-0.5 * ((spc[0]**2+spc[1]**2) / 0.4 ))
f = num/denom           


# Equation coefficients
K = Expression((("0.2","-0.16"),("-0.16","0.2")), degree = 0) # diffusion matrix
g = Constant(0.0) # Neumann bc

# Define boundary measure on Neumann part of boundary
dsN = Measure("ds", subdomain_id=1, subdomain_data=boundary_parts)


# Define trial and test function and solution at previous time-step
u = Function(V)
v = TestFunction(V)
u0 = Function(V)


# Time-stepping parameters
t_end = 1
dt = 0.001

# Define time discretized equation
F = inner(u0-u, v)*dx- 0.5*dt*inner(K*grad(u), grad(v))*dx + dt*f*v*dx -0.5 *dt*dot(nabla_grad(u),nabla_grad(u))*v*dx 

# Prepare solution function and solver
# Time-stepping
t = 1.0
u.rename("u", "temperature")
u0.vector()[:] = 50*np.sum((V.tabulate_dof_coordinates()-np.array([[1.5,1.5]]))**2,1)

# Open figure for plots
fig = plt.figure()
plt.show(block=False)

# save solution at each time step; including the final time
sol_array = np.array(u0.vector().get_local().reshape(ny+1,nx+1)).reshape(1,ny+1,nx+1)
while t >dt:

    # Solve the problem
    solve(F==0,u,
      solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})
    # plot solution
    p = plot(u, title='Solution at t = %g' % (t))
    if p is not None:
        plt.colorbar(p)
    fig.canvas.draw()
    plt.show()
    plt.clf()

    # Move to next time step
    u0.assign(u)
    t -= dt


    info('t = %g' % t)
    sol_array = np.append(sol_array,u0.vector().get_local().reshape(ny+1,nx+1).reshape(1,ny+1,nx+1),0)
    

dof_coord = V.tabulate_dof_coordinates()
np.save('dof_coord.npy',dof_coord)
np.save('sol_array.npy',sol_array)

    
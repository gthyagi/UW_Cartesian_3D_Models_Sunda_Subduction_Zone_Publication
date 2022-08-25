#!/usr/bin/env python
# coding: utf-8

# ### Subduction 3D with varying slab depth
# 
# 1. No post processing particles

# In[ ]:


import underworld as uw
import UWGeodynamics as GEO
import math
import numpy as np
from underworld import function as fn
import os
os.environ["UW_ENABLE_TIMING"] = "1"
import time
import h5py


# In[ ]:


if GEO.nProcs==1:
    from UWGeodynamics import visualisation as vis
plotting_on = False


# **Scaling of parameters**

# In[ ]:


rho_M             = 1.
g_M               = 1.
Height_M          = 1.
viscosity_M       = 1.


# In[ ]:


rho_N             = 50.0 # kg/m**3  note delta rho
g_N               = 9.81 # m/s**2
Height_N          = 1000e3 # m
viscosity_N       = 1e19 # Pa.sec or kg/m.sec


# In[ ]:


#Non-dimensional (scaling)
rho_scaling 		= rho_N/rho_M
viscosity_scaling 	= viscosity_N/viscosity_M
g_scaling 		= g_N/g_M
Height_scaling 		= Height_N/Height_M
pressure_scaling 	= rho_scaling * g_scaling * Height_scaling
time_scaling 		= viscosity_scaling/pressure_scaling
strainrate_scaling 	= 1./time_scaling
velocity_scaling        = Height_scaling/time_scaling
pressure_scaling_MPa    = rho_scaling * g_scaling * Height_scaling/1e6


# \begin{align}
# {\tau}_N = \frac{{\rho}_{0N}{g}_N{l}_N}{{\rho}_{0M}{g}_M{l}_M} {\tau}_M
# \end{align}
# 
# \begin{align}
# {V}_N = \frac{{\eta}_{0M}}{{\eta}_{0N}}\frac{{\rho}_{0N}{g}_N{{l}_N}^2}{{\rho}_{0M}{g}_M{{l}_M}^2} {V}_M
# \end{align}

# #### Cohesion Info

# In[ ]:


coh_dim = 30
coh_nd = np.round(coh_dim/pressure_scaling_MPa, 3)
# print(coh_nd)


# In[ ]:


# solver options and settings
"""
solver: default fgmres, mg, mumps, slud (superludist), lu (only serial)
"""
solver 		= 'fgmres'
inner_rtol 	= 1e-4   # def = 1e-5
outer_rtol 	= 1e-3
penalty_mg 	= 1.0e2
penalty_mumps 	= 1.0e6


# In[ ]:


#output directory string
'''
tao_Y1: const_coh (C, constant cohesion in the crust)
tao_Y2: coh_mu_rho_g_z (C + mu_rho_g*depth, depth dependent yield stress)
tao_Y3: coh_mu_eff_rho_g_z (C + mu_eff*rho_g*depth, velocity weaking mu)
'''
tao_Y_OC = 'const_coh'


# #### Resolution Info

# In[ ]:


resX = 512
resY = 512
resZ = 128
res  = str(resX)+str(resY)+str(resZ)


# In[ ]:


# creating output directory
outputPath = os.path.join(os.path.abspath("/scratch/n69/tg7098/"),'uw_subduction_3d_var_dep_upfix_no_pp_'+res+'_'+str(coh_dim)+'/')
if uw.mpi.rank ==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.mpi.barrier()


# #### Model Initiation

# In[ ]:


mesh          = uw.mesh.FeMesh_Cartesian( elementRes  = (resX, resY, resZ),
                                          minCoord    = (0., 0., 0.),
                                          maxCoord    = (4., 4., 1.) )
velocityField = mesh.add_variable( nodeDofCount=3 )
pressureField = mesh.subMesh.add_variable( nodeDofCount=1 )
strainRateInvField  = mesh.add_variable( nodeDofCount=1 )


# In[ ]:


# creating swarm to model subduction
swarm = uw.swarm.Swarm(mesh, particleEscape=True)
layout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm, particlesPerCell=20)
swarm.populate_using_layout(layout)
materialVariable = swarm.add_variable( dataType="int", count=1 )
pol_con = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=20)


# #### Create a 3D volume

# In[ ]:


def plot_particles():
    Fig = vis.Figure(resolution=(1200,600), axis=True)
    Fig.Points(swarm, materialVariable, cullface=False, opacity=1., colours='white green red purple blue', discrete=True)
    Fig.Mesh(mesh)
    lv = Fig.window()
    lv.rotate('z', 0)
    lv.rotate('x', -90)
    lv.redisplay()
    return


# In[ ]:


# material indices
UMantleIndex 	     = 0
UPCrustIndex 	     = 1
UPMantleIndex 	     = 2
SPCrustIndex 	     = 3
SPMantleIndex 	     = 4
SCrustLongIndex      = 5
SMantleLongIndex     = 6
SCrustShortIndex     = 7
SMantleShortIndex    = 8
LMantleIndex         = 9


# In[ ]:


# depth of the long and short slab
dep_long_slab = 1 - (660./1000)
dep_short_slab = 1 - (330./1000)
UPCrust_thk = 30.
UPLitho_thk = 50.


# In[ ]:


# indexing the material variable
materialVariable.data[:] = UMantleIndex
# plot_particles()


# In[ ]:


lowerMantleShape = GEO.shapes.HalfSpace(normal=(0.,0.,1.), origin=(0., 0., (1-(660./1000))))
materialVariable.data[lowerMantleShape.evaluate(swarm)] = LMantleIndex
# plot_particles()


# #### Subducting Plate Volume

# ##### Subducting Plate Lithosphere (SPL)

# In[ ]:


SPL_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(1., 1., (1.-0.)))
SPL_east   = GEO.shapes.HalfSpace(normal=(1., 0., math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
SPL_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(1., 1., (1.-(100./1000.))))
SPL_west   = GEO.shapes.HalfSpace(normal=(-1.,0.,-math.cos(math.pi/4)), origin=(0., 1., (1.-0.)))
# SPL_north  = GEO.shapes.HalfSpace(normal=(0.,1.,0.), origin=(1000.*u.kilometer,2000.*u.kilometer,0.*u.kilometer))
SPL_CompositeShape = SPL_top & SPL_east & SPL_bottom & SPL_west #& SPL_north
materialVariable.data[SPL_CompositeShape.evaluate(swarm)] = SPMantleIndex
# plot_particles()


# #### Slab Volume reaching 660km

# ##### Slab Lithosphere (SL)

# In[ ]:


SL_long_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(2., 1., (1.-0.)))
SL_long_east   = GEO.shapes.HalfSpace(normal=(1., 0., math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
SL_long_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(2.2, 1., dep_long_slab))
SL_long_west   = GEO.shapes.HalfSpace(normal=(-1., 0., -math.cos(math.pi/4)), origin=(2., 1., (1.-(170./1000.))))
SL_long_north  = GEO.shapes.HalfSpace(normal=(0., 1., 0.), origin=(1., 2., (1.-0.)))
SL_long_CompositeShape = SL_long_top & SL_long_bottom & SL_long_east & SL_long_west & SL_long_north
materialVariable.data[SL_long_CompositeShape.evaluate(swarm)] = SMantleLongIndex
# plot_particles()


# ##### Slab Crust (SC)

# In[ ]:


SC_long_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(2., 1., (1.-0.)))
SC_long_east   = GEO.shapes.HalfSpace(normal=(1., 0., math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
SC_long_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(2.2, 1., dep_long_slab))
SC_long_west   = GEO.shapes.HalfSpace(normal=(-1., 0., -math.cos(math.pi/4)), origin=(2., 1., (1.-(52./1000.))))
SC_long_north  = GEO.shapes.HalfSpace(normal=(0., 1., 0.), origin=(1., 2., (1.-0.)))
SC_long_CompositeShape = SC_long_top & SC_long_bottom & SC_long_east & SC_long_west & SC_long_north
materialVariable.data[SC_long_CompositeShape.evaluate(swarm)] = SCrustLongIndex
# plot_particles()


# #### Slab Volume reaching 330km

# ##### Slab Lithosphere (SL)

# In[ ]:


SL_short_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(2., 1., (1.-0.)))
SL_short_east   = GEO.shapes.HalfSpace(normal=(1., 0., math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
SL_short_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(2.2, 1., dep_short_slab))
SL_short_west   = GEO.shapes.HalfSpace(normal=(-1., 0., -math.cos(math.pi/4)), origin=(2., 1., (1.-(170./1000.))))
SL_short_north  = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(1., 2., (1.-0.)))
SL_short_CompositeShape = SL_short_top & SL_short_bottom & SL_short_east & SL_short_west & SL_short_north
materialVariable.data[SL_short_CompositeShape.evaluate(swarm)] = SMantleShortIndex
# plot_particles()


# ##### Slab Crust (SC)

# In[ ]:


SC_short_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(2., 1., (1.-0.)))
SC_short_east   = GEO.shapes.HalfSpace(normal=(1., 0., math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
SC_short_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(2.2, 1., dep_short_slab))
SC_short_west   = GEO.shapes.HalfSpace(normal=(-1., 0., -math.cos(math.pi/4)), origin=(2., 1., (1.-(52./1000.))))
SC_short_north  = GEO.shapes.HalfSpace(normal=(0., -1., 0.), origin=(1., 2., (1.-0.)))
SC_short_CompositeShape = SC_short_top & SC_short_bottom & SC_short_east & SC_short_west & SC_short_north
materialVariable.data[SC_short_CompositeShape.evaluate(swarm)] = SCrustShortIndex
# plot_particles()


# ##### Subducting Plate Crust (SPC)

# In[ ]:


SPC_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(1., 1., (1.-0.)))
SPC_east   = GEO.shapes.HalfSpace(normal=(1., 0., math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
SPC_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(1., 1., (1.-(30./1000.))))
SPC_west   = GEO.shapes.HalfSpace(normal=(-1., 0., -math.cos(math.pi/4)), origin=(0., 1., (1.-0.)))
SPC_CompositeShape = SPC_top & SPC_east & SPC_bottom & SPC_west
materialVariable.data[SPC_CompositeShape.evaluate(swarm)] = SPCrustIndex
# plot_particles()


# #### Upper Plate Volume

# ##### Upper Plate Lithosphere (UPL) (50 km)

# In[ ]:


UPL_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(3., 1., (1.-0.)))
UPL_east   = GEO.shapes.HalfSpace(normal=(1., 0., 0.), origin=(4., 1., (1.-0.)))
UPL_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(3., 1., (1.-(UPLitho_thk/1000.))))
UPL_west   = GEO.shapes.HalfSpace(normal=(-1., 0., -math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
UPL_CompositeShape = UPL_top & UPL_east & UPL_bottom & UPL_west 
materialVariable.data[UPL_CompositeShape.evaluate(swarm)] = UPMantleIndex
# plot_particles()


# ##### Upper Plate Crust (UPC) (30 km)

# In[ ]:


UPC_top    = GEO.shapes.HalfSpace(normal=(0., 0., 1.), origin=(3., 1., (1.-0.)))
UPC_east   = GEO.shapes.HalfSpace(normal=(1., 0., 0.), origin=(4., 1., (1.-0.)))
UPC_bottom = GEO.shapes.HalfSpace(normal=(0., 0., -1.), origin=(3., 1., (1.-(UPCrust_thk/1000.))))
UPC_west   = GEO.shapes.HalfSpace(normal=(-1., 0., -math.cos(math.pi/4)), origin=(2., 1., (1.-0.)))
UPC_CompositeShape = UPC_top & UPC_east & UPC_bottom & UPC_west 
materialVariable.data[UPC_CompositeShape.evaluate(swarm)] = UPCrustIndex
# plot_particles()


# In[ ]:


# turn on visualization and save swarm and matvar 
if GEO.nProcs==1 and plotting_on:
    plot_particles()
    mesh.save('mesh.h5')
    swarm_h5 = swarm.save('swarm.h5')
    mat_Var_h5 = materialVariable.save('matVar.h5')
    materialVariable.xdmf('matVar.xdmf',mat_Var_h5,"materialVariable",swarm_h5,"swarm")


# **Viscosities**

# In[ ]:


UMantleViscosity 	= 1.0
UPCrustViscosity 	= 1000.0
UPMantleViscosity 	= 1000.0
SPCrustViscosity 	= 1000.0
SPMantleViscosity 	= 1000.0
SCrustLongViscosity     = 1000.0
SMantleLongViscosity    = 1000.0
SCrustShortViscosity    = 1000.0
SMantleShortViscosity   = 1000.0
LMantleViscosity        = 30.0

# The yielding of the upper slab is dependent on the strain rate.
strainRate_2ndInvariant = fn.tensor.second_invariant(fn.tensor.symmetric(velocityField.fn_gradient))


# In[ ]:


# rheology1: viscoplastic crust and rest is newtonian
coord = fn.input()

if tao_Y_OC == 'const_coh':
    tao_Y_slab 		= coh_nd
if tao_Y_OC == 'coh_mu_rho_g_z':
    mu_rho_g 		= mu*1.0*1. # mu = 0.6, rho = 3300, g = 9.81 
    tao_Y_slab 		= coh_nd  + mu_rho_g *(1. - fn.math.sqrt(coord[0]**2. + coord[1]**2.))
if tao_Y_OC == 'coh_mu_eff_rho_g_z':
    vc_crit 		= (4.4/(velocity_scaling))*(10e-2/(365*24*60*60)) # v_crit = 4.4 cm/yr
    vc_mag 		= fn.math.sqrt(fn.math.dot(vc,vc))
    mu_eff 		= 0.6*(1.-0.7) + 0.6*(0.7/(1.+(vc_mag/vc_crit))) # mu_s*(1-gamma) + mu_s*(gamma/(1+(v/vc)))
    rho_g 		= 1.0*1.0 # mu = 0.6, rho = 3300, g = 9.81 
    tao_Y_slab 		= coh_nd  + mu_eff*rho_g *(1. - fn.math.sqrt(coord[0]**2. + coord[1]**2.))
yielding_slab       = 0.5 * tao_Y_slab / (strainRate_2ndInvariant+1.0e-18)


# In[ ]:


# Rheology type
viscoplastic 	= True
Non_Newtonian 	= False

# Viscosity limiter
eta_min = UMantleViscosity
eta_max = SPCrustViscosity 

# All are Newtonian except viscoplastic oceanic crust
if viscoplastic:
    slabYieldvisc = fn.exception.SafeMaths(fn.misc.max(eta_min, fn.misc.min(SPCrustViscosity, yielding_slab)))
    
# Non-Newtonian and viscoplastic crust (~30km) and Newtonian mantle
if Non_Newtonian:
    n 			= 3.
    sr_T 		= 1e-4
    creep_dislocation 	= slabMantleViscosity * fn.math.pow(((strainRate_2ndInvariant+1.0e-18)/sr_T), (1.-n)/n)
    creep 		= fn.exception.SafeMaths(fn.misc.min(creep_dislocation,slabMantleViscosity))
    slabYieldvisc 	= fn.exception.SafeMaths(fn.misc.min(creep, yielding))


# In[ ]:


# Viscosity function for the materials 
viscosityMap = {UMantleIndex 	     : UMantleViscosity,
                UPCrustIndex 	     : UPCrustViscosity,
                UPMantleIndex 	     : UPMantleViscosity,
                SPCrustIndex 	     : slabYieldvisc,
                SPMantleIndex 	     : SPMantleViscosity,
                SCrustLongIndex      : slabYieldvisc,
                SMantleLongIndex     : SMantleLongViscosity,
                SCrustShortIndex     : slabYieldvisc,
                SMantleShortIndex    : SMantleShortViscosity,
                LMantleIndex         : LMantleViscosity}
viscosityFn = fn.branching.map( fn_key = materialVariable, mapping = viscosityMap )


# **Densities**

# In[ ]:


UMantleDensity 	       = 0.0
UPCrustDensity 	       = 0.0
UPMantleDensity        = 0.0
SPCrustDensity 	       = 1.0
SPMantleDensity        = 1.0
SCrustLongDensity      = 1.0
SMantleLongDensity     = 1.0
SCrustShortDensity     = 1.0
SMantleShortDensity    = 1.0
LMantleDensity         = 0.0


# In[ ]:


densityMap   = {UMantleIndex 	     : UMantleDensity,
                UPCrustIndex 	     : UPCrustDensity,
                UPMantleIndex 	     : UPMantleDensity,
                SPCrustIndex 	     : SPCrustDensity,
                SPMantleIndex 	     : SPMantleDensity,
                SCrustLongIndex      : SCrustLongDensity,
                SMantleLongIndex     : SMantleLongDensity,
                SCrustShortIndex     : SCrustShortDensity,
                SMantleShortIndex    : SMantleShortDensity,
                LMantleIndex         : LMantleDensity}
densityFn 	 = fn.branching.map( fn_key = materialVariable, mapping = densityMap )
z_hat        = ( 0.0, 0.0, 1.0 ) # Define our vertical unit vector using a python tuple
buoyancyFn   = -1.0 * densityFn * z_hat # now create a buoyancy force vector


# **Set Initial and Boundary Conditions**

# In[ ]:


# set initial conditions (and boundary values)
velocityField.data[:] = [0.,0., 0.]
pressureField.data[:] = 0.

# send boundary condition information to underworld
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
kWalls = mesh.specialSets["MinK_VertexSet"] + mesh.specialSets["MaxK_VertexSet"]

freeslipBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                               indexSetsPerDof = (iWalls, jWalls, kWalls) )


# **System Setup**

# In[ ]:


stokesSLE = uw.systems.Stokes( velocityField = velocityField, 
                               pressureField = pressureField,
#                                voronoi_swarm = swarm, 
                               conditions    = [freeslipBC,],
                               fn_viscosity  = viscosityFn, 
                               fn_bodyforce  = buoyancyFn )
stokesSolver = uw.systems.Solver(stokesSLE) # Create solver & solve


# In[ ]:


# inner solver type
if solver == 'fmgres':
    pass
if solver == 'lu':
    stokesSolver.set_inner_method("lu")
if solver == 'mumps':
    stokesSolver.set_penalty(penalty_mumps)
    stokesSolver.set_inner_method("mumps")
if solver == 'mg':
    stokesSolver.set_penalty(penalty_mg)
    stokesSolver.set_inner_method("mg")
#     stokesSolver.options.mg.levels = 6
if solver == 'slud':
    stokesSolver.set_inner_method('superludist')


# In[ ]:


# rtol value
if inner_rtol != 'def':
    stokesSolver.set_inner_rtol(inner_rtol)
    stokesSolver.set_outer_rtol(outer_rtol)


# In[ ]:


advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )


# **Analysis tools**

# In[ ]:


#The root mean square Velocity
velSquared 	= uw.utils.Integral( fn.math.dot(velocityField,velocityField), mesh )
area 		= uw.utils.Integral( 1., mesh )
Vrms 		= math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )


# **Update function**

# In[ ]:


# define an update function
def update():
    # Retrieve the maximum possible timestep for the advection system.
    dt = advector.get_max_dt()
    # Advect using this timestep size.
    advector.integrate(dt)
    return time+dt, step+1


# **Checkpointing function definition**

# In[ ]:


# variables in checkpoint function
# Creating viscosity Field
viscosityField       = mesh.add_variable(nodeDofCount=1)
viscosityVariable    = swarm.add_variable(dataType="float", count=1)

# creating material variable field
matVarField          = mesh.add_variable(nodeDofCount=1)

# Creating strain rate fn and fields
strainRateFn         = fn.tensor.symmetric(velocityField.fn_gradient)
strainRateDField     = mesh.add_variable(nodeDofCount=3)
strainRateNDField    = mesh.add_variable(nodeDofCount=3)
strainRateVariable   = swarm.add_variable(dataType="float", count=6) # strain rate variable

# Creating stress fn and fields
stressFn             = 2. * viscosityFn * strainRateFn
stressInvFn          = fn.tensor.second_invariant(stressFn)
stressInvVariable    = swarm.add_variable(dataType="float", count=1) # stress Inv variable
stressVariable       = swarm.add_variable(dataType="float", count=6) # stress variable

stressDField         = mesh.add_variable(nodeDofCount=3) # stress diagonal components
stressNDField        = mesh.add_variable(nodeDofCount=3) # stress non-diagonal components
stressInvField_sMesh = mesh.subMesh.add_variable(nodeDofCount=1) # creating field on submesh
# stressField_sMesh    = mesh.subMesh.add_variable(nodeDofCount=6) # stress components
stressField          = mesh.add_variable(nodeDofCount=6) # stress components

#creating density field
densityVariable      = swarm.add_variable("float", 1)
density_Field        = mesh.add_variable(nodeDofCount=1)


# In[ ]:


meshHnd = mesh.save(outputPath+'mesh.00000.h5')
def checkpoint():

    # save swarm and swarm variables
    swarmHnd            = swarm.save(outputPath+'swarm.'+str(step).zfill(5)+'.h5')
    materialVariableHnd = materialVariable.save(outputPath+'materialVariable.'+ str(step).zfill(5) +'.h5')

    # projecting matvar to mesh field
    matVar_projector    = uw.utils.MeshVariable_Projection(matVarField, materialVariable, type=0)
    matVar_projector.solve()
    matVarFieldHnd      = matVarField.save(outputPath+'matVarField.'+str(step).zfill(5)+'.h5')
    matVarField.xdmf(outputPath+'matVarField.'+str(step).zfill(5)+'.xdmf', matVarFieldHnd, "matVarField", meshHnd, "mesh", modeltime=time)

    # saving velocity field in xyz format
    velocityHnd     = velocityField.save(outputPath+'velocityField.'+str(step).zfill(5)+'.h5', meshHnd)
    velocityField.xdmf(outputPath+'velocityField.'+str(step).zfill(5)+'.xdmf', velocityHnd, "velocity", meshHnd, "mesh", modeltime=time)

    # saving pressure field
    pressureHnd     = pressureField.save(outputPath+'pressureField.'+str(step).zfill(5)+'.h5', meshHnd)
    pressureField.xdmf(outputPath+'pressureField.'+str(step).zfill(5)+'.xdmf', pressureHnd, "pressure", meshHnd, "mesh", modeltime=time)

    # saving strainrate invariant field
    strainRateInvField.data[:]  = strainRate_2ndInvariant.evaluate(mesh)[:]
    strainRateInvFieldHnd       = strainRateInvField.save(outputPath+'strainRateInvField.'+str(step).zfill(5)+'.h5', meshHnd)
    strainRateInvField.xdmf(outputPath+'strainRateInvField.'+str(step).zfill(5)+'.xdmf', strainRateInvFieldHnd, "strainRateInv", meshHnd, "mesh", modeltime=time)

#     # saving stress Inv on swarm
#     stressInvVariable.data[:]  = stressInvFn.evaluate(swarm)[:]
#     stressInvVariableHnd       = stressInvVariable.save(outputPath+'stressInvVariable.'+ str(step).zfill(5) +'.h5')
#     stressInvVariable.xdmf(outputPath+'stressInvVariable.'+str(step).zfill(5)+'.xdmf', stressInvVariableHnd,"stressInvVariable",swarmHnd,"swarm",modeltime=time)

    # saving viscosity variable (swarm) and field (mesh)
    viscosityVariable.data[:]   = viscosityFn.evaluate(swarm)[:]
    viscosityVariableHnd        = viscosityVariable.save(outputPath+'viscosityVariable.'+ str(step).zfill(5) +'.h5')
    visc_projector              = uw.utils.MeshVariable_Projection(viscosityField, viscosityVariable, type=0) # Project to meshfield
    visc_projector.solve()
    viscosityFieldHnd           = viscosityField.save(outputPath+'viscosityField.'+str(step).zfill(5)+'.h5')
    viscosityField.xdmf(outputPath+'viscosityField.'+str(step).zfill(5)+'.xdmf', viscosityFieldHnd, "viscosityField", meshHnd, "mesh", modeltime=time)

    # saving strain rate variable (swarm) and field (mesh)
    strainRateVariable.data[:]  = strainRateFn.evaluate(swarm)[:]
    strainRateVariableHnd       = strainRateVariable.save(outputPath+'strainRateVariable.'+ str(step).zfill(5) +'.h5')
    strainRateDField.data[:]    = strainRateFn.evaluate(mesh)[:,0:3]
    strainRateNDField.data[:]   = strainRateFn.evaluate(mesh)[:,3:6]
    strainRateDFieldHnd         = strainRateDField.save(outputPath+'strainRateDField.'+str(step).zfill(5)+'.h5')
    strainRateDField.xdmf(outputPath+'strainRateDField.'+str(step).zfill(5)+'.xdmf', strainRateDFieldHnd,"strainRateDField",meshHnd,"mesh",modeltime=time)
    strainRateNDFieldHnd        = strainRateNDField.save(outputPath+'strainRateNDField.'+ str(step).zfill(5) +'.h5')
    strainRateNDField.xdmf(outputPath+'strainRateNDField.'+str(step).zfill(5)+'.xdmf', strainRateNDFieldHnd,"strainRateNDField",meshHnd,"mesh",modeltime=time)
    
    # saving stress variable (swarm) and field (submesh)
    stressVariable.data[:]      = stressFn.evaluate(swarm)[:]
    stressVariableHnd           = stressVariable.save(outputPath+'stressVariable.'+ str(step).zfill(5) +'.h5')
#     stress_proj                 = uw.utils.MeshVariable_Projection(stressField_sMesh, stressVariable, voronoi_swarm=swarm, type=0)
#     stress_proj.solve()
#     stressDField.data[:]        = stressField_sMesh.evaluate(mesh)[:,0:3]
#     stressNDField.data[:]       = stressField_sMesh.evaluate(mesh)[:,3:6]
    stress_proj                 = uw.utils.MeshVariable_Projection(stressField, stressVariable, voronoi_swarm=swarm, type=0)
    stress_proj.solve()
    stressDField.data[:]        = stressField.evaluate(mesh)[:,0:3]
    stressNDField.data[:]       = stressField.evaluate(mesh)[:,3:6]
    stressDFieldHnd             = stressDField.save(outputPath+'stressDField.'+str(step).zfill(5)+'.h5')
    stressDField.xdmf(outputPath+'stressDField.'+str(step).zfill(5)+'.xdmf', stressDFieldHnd, "stressDField", meshHnd, "mesh", modeltime=time)
    stressNDFieldHnd            = stressNDField.save(outputPath+'stressNDField.'+str(step).zfill(5)+'.h5')
    stressNDField.xdmf(outputPath+'stressNDField.'+str(step).zfill(5)+'.xdmf', stressNDFieldHnd, "stressNDField", meshHnd, "mesh", modeltime=time)

    # saving stress Invariant on submesh
    stressInvVariable.data[:]   = stressInvFn.evaluate(swarm)[:]
    stress_Inv_proj             = uw.utils.MeshVariable_Projection(stressInvField_sMesh, stressInvVariable, voronoi_swarm=swarm, type=0)
    stress_Inv_proj.solve()
    stressInvField_sMeshHnd     = stressInvField_sMesh.save(outputPath+'stressInvField_sMesh.'+str(step).zfill(5)+'.h5')
    stressInvField_sMesh.xdmf(outputPath+'stressInvField_sMesh.'+str(step).zfill(5)+'.xdmf', stressInvField_sMeshHnd, "stressInvField_sMesh", meshHnd, "mesh", modeltime=time)


# **Main simulation loop**
# 
# The main time stepping loop begins here. Inside the time loop the velocity field is solved for via the Stokes system solver and then the swarm is advected using the advector integrator. Basic statistics are output to screen each timestep.
# 
# 

# In[ ]:


time                    = 0.  # Initial time
step                    = 0   # Initial timestep
maxSteps                = 2   # Maximum timesteps
steps_output            = 1   # output every 1 timesteps


# In[ ]:


while step < maxSteps:
    # Solve non linear Stokes system
    stokesSolver.solve(nonLinearIterate=True, print_stats=True, nonLinearMaxIterations=50)
    if step % steps_output == 0 or step == maxSteps-1: # output intervals
        pol_con.repopulate()
        checkpoint()
        Vrms = math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )
        if uw.mpi.rank==0:
            print ('step = {0:6d}; time = {1:.3e}; Vrms = {2:.3e}'.format(step,time,Vrms))
    # update
    time,step = update()


# In[ ]:


if uw.mpi.rank == 0:
    print("Inner (velocity) Solve Options:")
    stokesSolver.options.A11.list()
    print('----------------------------------')
    print("Outer Solve Options:")
    stokesSolver.options.scr.list()
    print('----------------------------------')
    print("Multigrid (where enabled) Options:")
    stokesSolver.options.mg.list()
    # print("Penalty for mg:", penalty_mg)


# In[ ]:





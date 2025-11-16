import numpy as np
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input, Mixture
from Dyer_N2O_func_output import *

### All units SI

### INPUTS

# Fluid types
N2O = Fluid(FluidsList.NitrousOxide)

ethanol = Fluid(FluidsList.Ethanol)

E85 = Mixture(
    fluids=[FluidsList.Ethanol, FluidsList.nHeptane],
    fractions=[85, 15]
)

# Oxidiser assignment
ox_fluid = N2O

# Fuel assignment
fuel_fluid = ethanol

OF_targ = 2

# Propellant tank internal dimensions
id_tank = 98e-3 # m
l_tank = 400e-3 # m
v_tank = l_tank*id_tank**2*np.pi/4 # m3
print(v_tank*1000)
LD_tank = l_tank/id_tank

# Propellant initial state
T_init = 0 # C
p_init = 35e5 # Pa

ox = ox_fluid.with_state(Input.temperature(T_init),Input.pressure(p_init))
fuel = fuel_fluid.with_state(Input.temperature(T_init),Input.pressure(p_init))

rho_ox = ox.density
rho_f = fuel.density

z_piston = l_tank * rho_f / (rho_ox/OF_targ + rho_f) # m, z pos of piston. 0 = no fuel

m_ox_i = rho_ox*(z_piston/l_tank)*v_tank
m_f_i = rho_f*(1-z_piston/l_tank)*v_tank

# Injector
inj_ox_A = 50e-6 # m2
inj_ox_Cd = 0.7 #
inj_f_A = 8e-6 # m2
inj_f_Cd = 0.7 #

# Nozzle
nozzle_A = 0

# Time stepping
dt = 0.1 # s
t_end = 20 # s
t = np.arange(0,t_end+dt,dt) # s
n = len(t)

# Initialise
p_c = np.ones((n))*2e5
p_s = np.full(n,np.nan)
p_s[0] = p_init

m_oxs = np.full(n,np.nan)
m_fs = np.full(n,np.nan)
m_oxs[0] = m_ox_i
m_fs[0] = m_f_i

### CALCULATION

dPs_Dyer, dm_Dyer = Dyer_N2O(T_init+273.15,inj_ox_Cd,inj_ox_A)

for i in range(n-1):
    dP = p_s[i] - p_c[i]
    dm_ox = np.interp(p_c[i], dPs_Dyer, dm_Dyer)
    dm_f = inj_f_Cd * inj_f_A * np.sqrt(2 * rho_f * p_c[i])

    m_oxs[i+1] = m_oxs[i] - dm_ox
    m_fs[i+1] = m_fs[i] - dm_f

    # Move piston
    v_f = m_fs[i]/rho_f
    z_piston = v_f/(id_tank*np.pi)

    # Update density
    v_ox = v_tank - v_f
    rho_ox = m_oxs[i]/v_ox

    # Update propellant state
    ox = ox_fluid.with_state(Input.temperature(T_init),Input.density(rho_ox))
    fuel = fuel_fluid.with_state(Input.temperature(T_init),Input.pressure(ox.pressure))

    p_s[i+1] = ox.pressure

    if m_oxs[i] <= 0 or m_fs[i] <= 0:
        break


### OUTPUT


### PLOTTING

plt.plot(t,m_oxs,label="ox")
plt.plot(t,m_fs,label="f")
plt.legend()
plt.show()

plt.figure()
plt.plot(t,p_s)
plt.show()


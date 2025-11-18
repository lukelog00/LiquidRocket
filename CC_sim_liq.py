import numpy as np
import matplotlib.pyplot as plt
from pyfluids import Fluid, FluidsList, Input, Mixture
from Dyer_N2O_func_output import *
from rocketcea.cea_obj_w_units import CEA_Obj
import scipy

### All units SI

### TODO

# Model evaporation from oxidiser volume increase
# 
# 

### INPUTS

# Constants

g = 9.81 # m/s2
p_SSL = 1.013e5 # Pa
Ru = 8.314462618 # J/(mol.K)

# Fluid types
N2O = Fluid(FluidsList.NitrousOxide)

ethanol = Fluid(FluidsList.Ethanol)

E85 = Mixture(
    fluids=[FluidsList.Ethanol, FluidsList.nHeptane],
    fractions=[85, 15]
)

# Oxidiser assignment
ox_fluid = N2O
ox_type = "N2O"

# Fuel assignment
fuel_fluid = ethanol
fuel_type = "Ethanol"

cea = CEA_Obj(oxName=ox_type, fuelName=fuel_type)

OF_targ = 3

# Propellant tank internal dimensions
ullage = 0.05
id_tank = 72.2e-3 # m
l_tank = 210e-3 # m
A_tank = np.pi * id_tank ** 2 / 4
v_tank = l_tank*id_tank**2*np.pi/4 # m3
LD_tank = l_tank/id_tank
print("Tank L/D:",round(LD_tank,2))

# Propellant initial state
p_init = 30e5 # Pa

# Injector
inj_ox_d = 2.5e-3 # m
inj_ox_n = 12 #
inj_ox_A = np.pi * inj_ox_d ** 2 / 4 * inj_ox_n # m2
inj_ox_Cd = 0.7 #

inj_f_d = 1e-3 # m
inj_f_n = 3 #
inj_f_A = np.pi * inj_f_d ** 2 / 4 * inj_f_n # m2
inj_f_Cd = 0.7 #

# Nozzle
d_t = 30e-3 # m
At = np.pi * d_t ** 2 / 4 #100e-6 # m2
AeAt = 3.5

# Time stepping
dt = 0.01 # s
t_end = 20 # s
t = np.arange(0,t_end+dt,dt) # s
n = len(t)

# Initialise
i = 0

# State
ox = ox_fluid.with_state(Input.quality(ullage),Input.pressure(p_init))

T_init = ox.temperature + 1

fuel = fuel_fluid.with_state(Input.temperature(T_init),Input.pressure(p_init))

rho_ox = ox.density
rho_f = fuel.density

l_ullage = l_tank * ullage
l_use = l_tank - l_ullage
z_piston = l_use * rho_f / (rho_ox/OF_targ + rho_f) # m, z pos of piston. 0 = no oxidiser

v_ox_i = z_piston * A_tank
v_ox_v = l_ullage * A_tank
v_f_i = v_tank - v_ox_v - v_ox_i

m_ox_i = rho_ox*v_ox_i
m_f_i = rho_f*v_f_i

# Ullage mass
m_ox_v = ullage * m_ox_i

# Arrays
p_c = np.full(n,np.nan)
p_c[0] = p_SSL
p_s = np.full(n,np.nan)
p_s[0] = p_init

m_oxs = np.full(n,np.nan)
m_fs = np.full(n,np.nan)
m_oxs[0] = m_ox_i
m_fs[0] = m_f_i

OFs = np.full(n,np.nan)

cstars = np.full(n,np.nan)
dm_noz = np.full(n,np.nan)
ox_qual = np.full(n,np.nan)
Isp = np.full(n,np.nan)

Ts = np.full(n,np.nan)

### FUNCTIONS

# Dyer lookup data for mass flow vs pressure for injector geom
dPs_Dyer, dm_Dyer = Dyer_N2O(T_init+273.15,inj_ox_Cd,inj_ox_A)

# Formula for change in mass in chamber
def dm_c(p_c_loc):
    dP_i = max(p_s[i]-p_c_loc,0)

    dm_ox = np.interp(dP_i, dPs_Dyer, dm_Dyer)*dt # Ox mass through inj, kg
    dm_f = inj_f_Cd * inj_f_A * np.sqrt(2 * rho_f * dP_i)*dt # Fuel mass through inj, kg

    dm_noz = (p_c_loc-p_SSL) * At / cstars[i] * dt # Mass through nozzle, kg
    return (dm_ox + dm_f - dm_noz)


### CALCULATION

for i in range(n-1):
    dP_i = p_s[i] - p_c[i]
    dm_ox = np.interp(abs(dP_i), dPs_Dyer, dm_Dyer)*dt # Ox mass through inj, kg
    dm_f = inj_f_Cd * inj_f_A * np.sqrt(2 * rho_f * abs(dP_i))*dt # Fuel mass through inj, kg

    if dP_i < 0:
        print("Reverse flow detected, this could just mean the burn is finished")
        break

    OFs[i] = dm_ox/dm_f
    ox_qual[i] = ox.quality

    # CEA
    cstars[i] = cea.get_Cstar(Pc=p_c[i]/1e5, MR=OFs[i])
    T0,Tt,Te = cea.get_Temperatures(Pc=p_c[i]/1e5, MR=OFs[i])
    mw, gamma = cea.get_Chamber_MolWt_gamma(Pc=p_c[i]/1e5, MR=OFs[i])
    R = Ru / mw

    dm_noz[i] = (p_c[i] - p_SSL) * At / cstars[i] # Mass flow rate through nozzle, kg

    # Exit mach number
    Me = cea.get_MachNumber(Pc=p_c[i]/1e5, MR=OFs[i],eps=AeAt)

    # Exit speed
    Ve = Me * np.sqrt(gamma * R * Te)

    # Thrust
    Cfcea, Cfamb, mode = cea.get_PambCf(Pamb=p_SSL/1e5, Pc=p_c[i]/1e5, MR=OFs[i],eps=AeAt)
    Ts[i] = Cfamb*p_c[i]*At # Thrust at given ambient pressure

    # Isp
    IspVac = cea.get_Isp(Pc=p_c[i]/1e5, MR=OFs[i],eps=AeAt)
    Isp[i], mode = cea.estimate_Ambient_Isp(Pamb=p_SSL/1e5, Pc=p_c[i]/1e5, MR=OFs[i],eps=AeAt)

    # Update propellant masses
    m_oxs[i+1] = m_oxs[i] - dm_ox
    m_fs[i+1] = m_fs[i] - dm_f

    # Update chamber pressure
    # Check that we can do root finding on chamber pressure:
    # If residual mass flows are both positive
    if dm_c(p_init) > 0 and dm_c(p_SSL) > 0:
        print("Nozzle throat too small or injector area too large!")
        break
    # If residual mass flows are both negative:
    elif dm_c(p_init) < 0 and dm_c(p_SSL) < 0:
        print("Nozzle throat too large or injector area too small!")
        break
    # Solve for p_c where dm_chamber = 0
    p_c[i+1] = scipy.optimize.brentq(dm_c,p_SSL,p_init)

    # Move piston
    v_f = m_fs[i]/rho_f
    z_piston = v_f/(id_tank*np.pi)

    # Update nitrous supply density
    v_ox = v_tank - v_f
    rho_ox = m_oxs[i]/v_ox

    # Update propellant state
    ox = ox_fluid.with_state(Input.temperature(T_init),Input.density(rho_ox))
    fuel = fuel_fluid.with_state(Input.temperature(T_init),Input.pressure(ox.pressure))

    p_s[i+1] = ox.pressure

    if m_oxs[i] <= 0 or m_fs[i] <= 0:
        break


### OUTPUT

impulse = scipy.integrate.trapezoid(Ts[~np.isnan(Ts)],x=t[~np.isnan(Ts)])

print("Nozzle mode:",mode)
print("Impulse:",round(impulse),"Ns")
print("Thrust:",round(Ts[3]),"N")
print("Isp:",round(np.mean(Isp[~np.isnan(Isp)]),1),"s")

### PLOTTING

plt.figure()
plt.plot(t,m_oxs,label="ox")
plt.plot(t,m_fs,label="f")
plt.xlabel("Time (s)")
plt.ylabel("Mass (kg)")
plt.legend()

plt.figure()
plt.plot(t,ox_qual)
plt.xlabel("Time (s)")
plt.ylabel("Oxidiser vapor quality (%)")

plt.figure()
plt.plot(t,Ts)
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")

plt.figure()
plt.plot(t,p_s/1e5,label="supply")
plt.plot(t,p_c/1e5,label="chamber")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (bar)")
plt.legend()

plt.figure()
plt.plot(t,OFs)
plt.xlabel("Time (s)")
plt.ylabel("OF ratio")

# plt.figure()
# plt.plot(t,Isp)
# plt.xlabel("Time (s)")
# plt.ylabel("Isp (s)")

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.ticker as ticker
from pandas import read_csv

#Conversion in IU

G_p = 6.67259e-8 #G in cgs
M_sun = 1.9891e33 #solar mass in g
R_sun = 6.9598e10 #solar radius in cm 
year = 3.14159e7
ly = 9.463e17 #light year in cm
parsec = 3.086e18 #parsec in cm
AU = 1.496e13 #astronomical unit in cm

def v_IU(v_p, M_p=M_sun, r_p=AU):
    return np.sqrt(r_p / (G_p * M_p)) * v_p

def t_IU(t_p, M_p=M_sun, r_p=AU):
    return t_p / (np.sqrt(r_p / (G_p * M_p)) * r_p)

t_iu_yr = t_IU(year) #1 yr is 6.251839 IU
v_iu_cgs = v_IU(1) #1 cm/s is 3.357e-7 IU

#%%

#Simulation properties

N = 60 #Number of particles
M = 10 #Total mass in solar masses
a = 1 #Total radius in AU
m_i = M / N #Particle mass in solar masses

#%%

#Function that reads the simulation data from the output file

def GetData(filename, N):
    #Get data in a pandas dataframe
    data = read_csv(filename, 
                    names=["m", "x", "y", "z", "vx", "vy", "vz"], 
                    sep=" ", 
                    )
    
    #Get the simulation time (first column of the corresponding rows)
    time = np.array(data[1::N+2])[:, 0]

    #Remove the rows corresponding to N and time (which are padded with NaNs)
    data = data.dropna().reset_index()
    
    #Get the particle coordinates and velocities in the external frame
    #Rows are the evolution in time of a given particle, columns are the particle
    #number, and the third dimension are its coordinates/velocities
    pos_ext = np.array([data[["x", "y", "z"]][i::N] for i in range(N)]).transpose(1, 0, 2)
    vel_ext = np.array([data[["vx", "vy", "vz"]][i::N] for i in range(N)]).transpose(1, 0, 2)
    
    #Compute CM position ad velocity
    m = data["m"][0]
    CM_p = m * np.sum(pos_ext, axis=1) / (m*N)
    CM_v = m * np.sum(vel_ext, axis=1) / (m*N)
    
    #Convert from external to CM frame
    pos_CM = pos_ext.copy()
    vel_CM = vel_ext.copy()
    
    for i in range(CM_p.shape[0]):
        pos_CM[i, :, :] -= CM_p[i, :]
        vel_CM[i, :, :] -= CM_v[i, :]
    
    return time, m, pos_ext, vel_ext, pos_CM, vel_CM, CM_p, CM_v

#%%

#Get the simulation time and the positions and velocities of the particles

sim_file = "../DirectNBody/output_sphere_" + str(N) + ".out"

time, m_i, pos_ext, vel_ext, pos_CM, vel_CM, CM_p, CM_v = GetData(sim_file, N)

#Time in years
time_yr = time / t_iu_yr

#Positions and velocities of the particles in the external frame
x_ext, y_ext, z_ext = pos_ext[:, :, 0], pos_ext[:, :, 1], pos_ext[:, :, 2]
vx_ext, vy_ext, vz_ext = vel_ext[:, :, 0], vel_ext[:, :, 1], vel_ext[:, :, 2]

R = np.sqrt(x_ext**2 + y_ext**2 + z_ext**2)
V = np.sqrt(vx_ext**2 + vy_ext**2 + vz_ext**2)

#Positions and velocities of the particles in the CM frame
x_CM, y_CM, z_CM = pos_CM[:, :, 0], pos_CM[:, :, 1], pos_CM[:, :, 2]
vx_CM, vy_CM, vz_CM = vel_CM[:, :, 0], vel_CM[:, :, 1], vel_CM[:, :, 2]

R_CM = np.sqrt(x_CM**2 + y_CM**2 + z_CM**2)
V_CM = np.sqrt(vx_CM**2 + vy_CM**2 + vz_CM**2)

#Position and velocity of CM in the external frame
CM_x, CM_y, CM_z = CM_p[:, 0], CM_p[:, 1], CM_p[:, 2]
CM_vx, CM_vy, CM_vz = CM_v[:, 0], CM_v[:, 1], CM_v[:, 2]

CM_R = np.sqrt(CM_x**2 + CM_y**2 + CM_z**2)
CM_V = np.sqrt(CM_vx**2 + CM_vy**2 + CM_vz**2)

#%%

#Compute the dynamical time and the collapse time in years

density_0 = m_i * N * M_sun / (4/3 * np.pi * (a * AU)**3)

t_dyn = np.sqrt(3 * np.pi / (16 * G_p * density_0)) / year

t_coll = t_dyn / np.sqrt(2)

print("Dynamical time: t_dyn =" + f"{t_dyn: .3f}" + " yr")
print("Collapse time: t_coll =" + f"{t_coll: .3f}" + " yr")
print("Collapse time (IU) = " + str(t_coll * t_iu_yr))
print("Initial density = " + f"{density_0: .3e}" + " g cm^-3")

#%%

#Compute the total potential and kinetic energy of the system

K_tot = 0.5 * (m_i * M_sun) * np.sum((V / v_iu_cgs)**2, axis=1)
U_tot = np.zeros(len(time))

for i in range(N):
    for j in range(N):
        if j > i:
            U_tot += -G_p * (m_i * M_sun)**2 * np.sqrt(np.sum(AU**2 * (pos_ext[:, i, :] - pos_ext[:, j, :])**2, axis=1))**-1
            
E_tot = K_tot + U_tot

#%%

#Collapse animation

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.set_aspect("equal")

ticks = np.linspace(-2, 2., 5)
bbox = dict(boxstyle='round', fc='white', ec='black', alpha=0.5)

def update_collapse(frame):
    ax.clear()
    
    ax.set_autoscale_on(False)
    ax.scatter(x_ext[frame], y_ext[frame], z_ext[frame], color="darkcyan")
    ax.scatter(CM_x[frame], CM_y[frame], CM_z[frame], color="crimson")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xlabel("x [AU]", size=10)
    ax.set_ylabel("y [AU]", size=10)
    ax.set_zlabel("z [AU]", size=10)
    ax.text(0.875, 0.04, 0., s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox, color="black", size=10, transform=ax.transAxes)
    
end = np.where(time == max(time))[0][0]
animation = anim.FuncAnimation(fig=fig, 
                               func=update_collapse, 
                               frames=end, 
                               interval=20, repeat=True)

#%%

#Phase space visualization

fig_ph = plt.figure()
ax_ph = fig_ph.add_subplot()
ax_ph.set_aspect("equal")

ticks_x_ph = np.linspace(-2, 20., 10)
ticks_y_ph = np.linspace(-2, 25., 5)
bbox_ph = dict(boxstyle='round', fc='white', ec='black', alpha=0.8)

def update_ph(frame):
    ax_ph.clear()
    
    ax_ph.scatter(R[frame], V[frame], color="darkcyan", alpha=0.5)
    ax_ph.scatter(CM_R[frame], CM_V[frame], color="crimson")
    ax_ph.set_xticks(ticks_x_ph)
    ax_ph.set_yticks(ticks_y_ph)
    ax_ph.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax_ph.set_xlabel("R [AU]", size=10)
    ax_ph.set_ylabel("V [IU]", size=10)
    ax_ph.text(0.7, 0.03, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_ph, color="black", size=10, transform=ax_ph.transAxes)
    
end_ph = np.where(time == max(time))[0][0]
animation_ph = anim.FuncAnimation(fig=fig_ph, 
                               func=update_ph, 
                               frames=end_ph, 
                               interval=60, repeat=True)

#%%

#Total energy plot

plt.figure()
plt.plot(time_yr, K_tot, color="crimson", label="Total kinetic energy")
plt.plot(time_yr, U_tot, color="darkcyan", label="Total potential energy")
plt.plot(time_yr, E_tot, color="green", label="Total energy")
plt.xlabel("$t\ [yr]$")
plt.ylabel("$E\ [erg]$")
plt.legend()


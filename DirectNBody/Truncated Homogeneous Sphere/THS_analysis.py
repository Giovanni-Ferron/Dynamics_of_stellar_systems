import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from pandas import read_csv
import time as tm

#To view the animations type %matplotlib qt6 in the console

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

#%%----Simulation properties----

N = 800 #Number of particles
M = 10 #Total mass in solar masses
a = 1 #Total radius in AU
m_i = M / N #Particle mass in solar masses

#%%----GetData function----

#Function that reads the simulation data from the output file

def GetData(filename, N):
    #Get data in a pandas dataframe
    data = read_csv(filename, 
                    names=["m", "x", "y", "z", "vx", "vy", "vz"], 
                    sep=" ")
    
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

#%%----Read output file----

#Get the simulation time and the positions and velocities of the particles

sim_file = "Output/THS_N" + str(N) + "_M" + str(M) + "_a" + str(a) + ".out"

time, m_i, pos_ext, vel_ext, pos_CM, vel_CM, CM_p, CM_v = GetData(sim_file, N)

#Time in years
time_yr = time / t_iu_yr

#Positions and velocities of the particles in the external frame
x_ext, y_ext, z_ext = pos_ext[:, :, 0], pos_ext[:, :, 1], pos_ext[:, :, 2]
vx_ext, vy_ext, vz_ext = vel_ext[:, :, 0], vel_ext[:, :, 1], vel_ext[:, :, 2]

R = np.sqrt(np.sum(pos_ext**2, axis=2))
V = np.sqrt(np.sum(vel_ext**2, axis=2))
            
#Positions and velocities of the particles in the CM frame
x_CM, y_CM, z_CM = pos_CM[:, :, 0], pos_CM[:, :, 1], pos_CM[:, :, 2]
vx_CM, vy_CM, vz_CM = vel_CM[:, :, 0], vel_CM[:, :, 1], vel_CM[:, :, 2]

R_CM = np.sqrt(np.sum(pos_CM**2, axis=2))
V_CM = np.sqrt(np.sum(vel_CM**2, axis=2))

#Position and velocity of CM in the external frame
CM_x, CM_y, CM_z = CM_p[:, 0], CM_p[:, 1], CM_p[:, 2]
CM_vx, CM_vy, CM_vz = CM_v[:, 0], CM_v[:, 1], CM_v[:, 2]

CM_R = np.sqrt(np.sum(CM_p**2, axis=1))
CM_V = np.sqrt(np.sum(CM_v**2, axis=1))

#%%----Compute total energy in the CM frame----

#Compute the total potential and kinetic energy of the system in the CM frame

K_tot = 0.5 * (m_i * M_sun) * np.sum((V_CM / v_iu_cgs)**2, axis=1)
U_tot = np.zeros(len(time))

for i in range(N):
    for j in range(N):
        if j > i:
            U_tot += -G_p * (m_i * M_sun)**2 * np.sqrt(np.sum(AU**2 * (pos_CM[:, i, :] - pos_CM[:, j, :])**2, axis=1))**-1
            
E_tot = K_tot + U_tot

plt.figure()
plt.plot(time_yr, K_tot, color="crimson", label="Total kinetic energy")
plt.plot(time_yr, U_tot, color="darkcyan", label="Total potential energy")
plt.plot(time_yr, E_tot, color="green", label="Total energy")
plt.xlabel("$t\ [yr]$")
plt.ylabel("$E\ [erg]$")
plt.legend()

#%%----Compute total linear and angular momentum in the external frame----

p_tot = m_i * M_sun * np.sqrt(np.sum(vx_ext, axis=1)**2 + 
                              np.sum(vy_ext, axis=1)**2 + 
                              np.sum(vz_ext, axis=1)**2) / v_iu_cgs
L_tot = m_i * M_sun * AU * v_iu_cgs**-1 * np.sqrt(np.sum(y_ext * vy_ext - z_ext * vz_ext, axis=1)**2 + 
                                  np.sum(z_ext * vz_ext - x_ext * vx_ext, axis=1)**2 + 
                                  np.sum(x_ext * vx_ext - y_ext * vy_ext, axis=1)**2)

fig_p, ax_p = plt.subplots(1, 2, figsize=(10, 5))
ax_p[0].plot(time_yr, p_tot, color="darkcyan")
ax_p[0].set_xlabel("$t\ [yr]$")
ax_p[0].set_ylabel("$p\ [g\cdot cm\cdot s^{-1}]$")

ax_p[1].plot(time_yr, L_tot, color="darkcyan")
ax_p[1].set_xlabel("$t\ [yr]$")
ax_p[1].set_ylabel("$L\ [g\cdot cm^{2}\cdot s^{-1}]$")

#%%----Compute theoretical timescales----

#Compute the theoretical dynamical time and the collapse time in years

density_0 = m_i * N * M_sun / (4/3 * np.pi * (a * AU)**3)

t_dyn = np.sqrt(3 * np.pi / (16 * G_p * density_0)) / year

t_coll = t_dyn / np.sqrt(2)

print("Dynamical time: t_dyn =" + f"{t_dyn: .3f}" + " yr =" + f"{t_dyn * t_iu_yr: .3f}" + " IU")
print("Collapse time: t_coll =" + f"{t_coll: .3f}" + " yr =" + f"{t_coll * t_iu_yr: .3f}" + " IU")
print("Initial density = " + f"{density_0: .3e}" + " g cm^-3")

#%%----Compute simulation timescales----
#%%%Compute the collapse time from the simulation data

#Index of the particle with the farthest position from the CM at initial time
initial_R_idx = np.where(R_CM[0, :] == np.max(R_CM[0, :]))[0][0]

#Index of the minimum value in time of the radius of the farthest particle
min_R_idx = np.where(R_CM[:, initial_R_idx] == np.min(R_CM[:, initial_R_idx]))[0][0]

#Collapse time from the simulation
t_coll_sim = time[min_R_idx]

print("Collapse time from simulation =" + f"{t_coll_sim / t_iu_yr: .3f}" + " yr")

#%%%Compute the collapse time from the point of minimum potential energy

t_coll_en = time[U_tot == np.min(U_tot)][0]

print("Collapse time from minimum U_tot =" + f"{t_coll_en / t_iu_yr: .3f}" + " yr")

#%%----Collapse animation----

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.set_aspect("equal")

ticks = np.linspace(-2, 2., 5)
bbox = dict(boxstyle='round', fc='white', ec='black', alpha=0.5)

def update_collapse(frame):
    ax.clear()
    
    ax.set_autoscale_on(False)
    ax.scatter(x_CM[frame], y_CM[frame], z_CM[frame], color="darkcyan")
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

#%%----Phase space and position histogram----

#Phase space visualization and particle position histogram

fig_ph = plt.figure(figsize=(15, 5))

#Phase space plot

ax_ph = fig_ph.add_subplot(121)    

ticks_x_ph = np.linspace(-1, 5, 10)
ticks_y_ph = np.linspace(-1, 1.2 * int(np.max(V_CM)), 5)
bbox_ph = dict(boxstyle='round', fc='white', ec='black', alpha=0.8)

#Position histogram plot

ax_hist = fig_ph.add_subplot(122)

ticks_x_hist = np.linspace(-1, 0.8 * int(np.max(R_CM)), 10)
ticks_y_hist = np.linspace(0, N/1.5, 5)

def update_ph(frame):
    ax_ph.clear()
    
    ax_ph.scatter(R_CM[frame, :], V_CM[frame, :], color="darkcyan", alpha=0.5)
    ax_ph.scatter(CM_R[frame], CM_V[frame], color="crimson")
    ax_ph.set_xticks(ticks_x_ph)
    ax_ph.set_yticks(ticks_y_ph)
    ax_ph.set_xlabel("R [AU]", size=10)
    ax_ph.set_ylabel("V [IU]", size=10)   
    ax_ph.text(0.7, 0.03, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_ph, color="black", size=10, transform=ax_ph.transAxes)
    
    ax_hist.clear()
    
    ax_hist.hist(R_CM[frame, :], color="darkcyan")
    ax_hist.set_xticks(ticks_x_hist)
    ax_hist.set_yticks(ticks_y_hist)
    ax_hist.set_xlabel("R [AU]", size=10)
    
    
end_ph = np.where(time == max(time))[0][0]
animation_ph = anim.FuncAnimation(fig=fig_ph, 
                               func=update_ph, 
                               frames=end_ph, 
                               interval=60, repeat=True)

#%%----Plot the distribution density over time----

#Divide the sphere in volume elements, all with the same total volume
r_grid_len = 500
volume = 4/3 * np.pi * a**3 / r_grid_len

#Compute the radii of each volume element to form a grid of radii, adding a last
r_grid = np.zeros(r_grid_len)

for i in range(r_grid_len - 1):
    r_grid[i + 1] = (3/4 / np.pi * volume + r_grid[i]**3)**(1/3)
    
if r_grid[-1] < a:
    r_grid = np.append(r_grid, a)

#Compute the number of particles inside a given radius interval    
p_num = np.zeros((len(time), len(r_grid) - 1))

for i in range(0, len(time)):
    #Count the number of particles inside every radius interval at a given time
    p_idx, p_inside = np.unique(np.digitize(R_CM[i, :], r_grid), return_counts=True)
    
    #Count only the particles within a, and add the number to the p_num
    #elements at the index of the corresponding occupied interval, leaving
    #the unoccupied ones at zero particles    
    cond_within = np.where(p_idx < len(r_grid) - 1)
    p_num[i, p_idx[cond_within]] += p_inside[cond_within]

#Compute the density in each radius interval and the Poisson error
rho = (p_num / volume) * m_i * M_sun * AU**-3
rho_err = (np.sqrt(p_num) / volume) * m_i * M_sun * AU**-3

#%%%Density profile animation

#Plot the density at each radius corresponding to the middle of a bin
bar_pos = np.diff(r_grid) / 2 + r_grid[:-1]

fig_rho, ax_rho = plt.subplots()
bbox_rho = dict(boxstyle='round', fc='white', ec='black', alpha=0.4)

def update_rho(frame):
    ax_rho.clear()
    
    ax_rho.plot(bar_pos, rho[frame, :], color="darkcyan", label="Density profile")
    ax_rho.axhline(density_0, color="crimson", label="Analytical density profile $\\rho_0$")
    ax_rho.fill_between(bar_pos, 
                      rho[frame, :] - rho_err[frame, :], 
                      rho[frame, :] + rho_err[frame, :], alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
    plt.xlabel("$R\ [AU]$")
    plt.ylabel("$\\rho\ [g\ cm^{-3}]$")
    plt.legend(loc="upper right")
    ax_rho.text(0.81, 0.72, s="t =" + f"{time_yr[frame]: .3f}" + " yr", bbox=bbox_rho, color="black", size=10, transform=ax_rho.transAxes)
    
end_rho = np.where(time == max(time))[0][0]
animation_rho = anim.FuncAnimation(fig=fig_rho, 
                               func=update_rho, 
                               frames=end_rho, 
                               interval=120, repeat=True)
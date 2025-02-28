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

#Number of particles
N = np.array([10, 20, 50, 100, 200, 500])

#Total mass in solar masses
M = np.array([10, 50, 100, 150, 200])

#Total radius in AU
a = np.array([1, 5, 8 ,12])

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


#Function to compute collapse time
    
def ComputeCollTime(N, M, a, t, R):
    
    #Theoretical collapse time
    density_0 = M * M_sun / (4/3 * np.pi * (a * AU)**3)
    t_dyn = np.sqrt(3 * np.pi / (16 * G_p * density_0)) / year
    t_coll = t_dyn / np.sqrt(2)
    
    #Simulation collpase time
    
    #Index of the particle with the farthest position from the CM at initial time
    initial_R_idx = np.where(R[0, :] == np.max(R[0, :]))[0][0]
    
    #Index of the minimum value in time of the radius of the farthest particle
    min_R_idx = np.where(R[:, initial_R_idx] == np.min(R[:, initial_R_idx]))[0][0]
    
    #Collapse time from the simulation
    t_coll_sim = t[min_R_idx] / t_iu_yr
    
    return t_coll_sim, t_coll

#%%

#Compute and plot the collapse time for different N and fixed M and a

t_coll_sim = np.zeros(len(N))
t_coll_th = np.zeros(len(N))

for i, p_num in enumerate(N):
    filename = "Output/THS_N" + str(p_num) + "_M10_a1.out"
    time, m, pos_ext, vel_ext, pos_CM, vel_CM, CM_p, CM_v = GetData(filename, p_num)
    R_CM = np.sqrt(np.sum(pos_CM**2, axis=2))
    
    t_coll_sim[i], t_coll_th[i] = ComputeCollTime(p_num, 10, 1, time, R_CM)  
    
    
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(0, len(N)), t_coll_sim, color="darkcyan", marker="o", label="Simulation $t_{coll}$")    
ax.plot(np.arange(0, len(N)), t_coll_th, color="crimson", marker="o", label="Theoretical $t_{coll}$")
ax.set_xlabel("$N$")
ax.set_ylabel("$t_{coll}$")
ax.set_xticks(np.arange(0, len(N)))
ax.set_xticklabels(N)
ax.legend()

# CM_V = np.sqrt(CM_vx**2 + CM_vy**2 + CM_vz**2)
#%%

# #Compute the total potential and kinetic energy of the system in the CM frame

# K_tot = 0.5 * (m_i * M_sun) * np.sum((V_CM / v_iu_cgs)**2, axis=1)
# U_tot = np.zeros(len(time))

# for i in range(N):
#     for j in range(N):
#         if j > i:
#             U_tot += -G_p * (m_i * M_sun)**2 * np.sqrt(np.sum(AU**2 * (pos_CM[:, i, :] - pos_CM[:, j, :])**2, axis=1))**-1
            
# E_tot = K_tot + U

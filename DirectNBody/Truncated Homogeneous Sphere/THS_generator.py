import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

N = 2000 #Number of particles
M = 10 #Total mass in solar masses
a = 1 #Total radius in AU
m_i = M / N #Particle mass in solar masses

#%%

#Function to generate a distribution of particles to input in the NBody code

def GenInput(N, m_i, a):
    
    #Cumulative distributions
    P_r = np.random.uniform(0, 1, N) 
    P_phi = np.random.uniform(0, 1, N)
    P_theta = np.random.uniform(0, 1, N)
    
    #Draw spherical coordinates based on their PDF using the inverse cumulative
    r = a * P_r**(1/3)
    phi = 2 * np.pi * P_phi
    theta = np.arccos(1 - 2 * P_theta)
    
    #Convert in cartesian coordinates
    x_0 = r * np.cos(phi) * np.sin(theta)
    y_0 = r * np.sin(phi) * np.sin(theta)
    z_0 = r * np.cos(theta) 
    
    #Write to file
    file_path = "Input/THS_N" + str(N) + "_M" + str(M) + "_a" + str(a) + ".in"
    
    with open(file_path, "w") as i_file:
        i_file.write(str(N) + "\n0\n")
        
        for i in tqdm(range(N)):
            i_file.write(str(m_i) + " " + 
                         str(x_0) + " " + 
                         str(y_0) + " " + 
                         str(z_0) + "\n")
        
    return x_0, y_0, z_0
    
#%%

#Generate input
x_0, y_0, z_0 = GenInput(N, m_i, a)

#%%

#Visualize the generated distribution

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.set_aspect("equal")
ax.scatter(x_0, y_0, z_0, alpha=0.8, color="darkcyan")
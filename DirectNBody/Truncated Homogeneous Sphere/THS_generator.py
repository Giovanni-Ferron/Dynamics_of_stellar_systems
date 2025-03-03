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

#%%----Simulation properties----

N = 10000 #Number of particles
M = 10 #Total mass in solar masses
a = 10 #Total radius in AU
m_i = M / N #Particle mass in solar masses

#%%----Compute theoretical timescales----

#Compute the dynamical time and the collapse time in years

density_0 = M * M_sun / (4/3 * np.pi * (a * AU)**3)

t_dyn = np.sqrt(3 * np.pi / (16 * G_p * density_0)) / year

t_coll = t_dyn / np.sqrt(2)

print("Dynamical time: t_dyn =" + f"{t_dyn: .3f}" + " yr =" + f"{t_dyn * t_iu_yr: .3f}" + " IU")
print("Collapse time: t_coll =" + f"{t_coll: .3f}" + " yr =" + f"{t_coll * t_iu_yr: .3f}" + " IU")
print("Initial density = " + f"{density_0: .3e}" + " g cm^-3")

#%%----GenInput function----

#Function to generate a distribution of particles to input in the NBody code

def GenInput(N, m_i, a, to_file=False):
    
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
    
    #Initial velocities
    vx_0 = np.zeros(N)
    vy_0 = np.zeros(N)
    vz_0 = np.zeros(N)
    
    #Write to file
    if to_file:
        file_path = "Input/THS_N" + str(N) + "_M" + str(M) + "_a" + str(a) + ".in"
        
        with open(file_path, "w") as i_file:
            i_file.write(str(N) + "\n0\n")
            
            for i in tqdm(range(N)):
                i_file.write(str(m_i) + " " + 
                             str(x_0[i]) + " " + str(y_0[i]) + " " + str(z_0[i]) + " " +
                             str(vx_0[i]) + " " + str(vy_0[i]) + " " + str(vz_0[i]) + "\n")
        
    return x_0, y_0, z_0, r, phi, theta
    
#%%----Generate input----

x_0, y_0, z_0, r, phi, theta = GenInput(N, m_i, a, False)

#%%----Visualize the generated distribution----

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.set_aspect("equal")
ax.scatter(x_0, y_0, z_0, alpha=0.8, color="darkcyan")

#%%----Plot the spherical coordinates distributions----

fig_sp, ax_sp = plt.subplots(1, 3, figsize=(10, 5))

ax_sp[0].hist(r, bins="fd", color="darkcyan")
ax_sp[0].set_xlabel("$r\ [AU]$")

ax_sp[1].hist(phi, bins="fd", color="darkcyan")
ax_sp[1].set_xlabel("$\phi\ [rad]$")

ax_sp[2].hist(theta, bins="fd", color="darkcyan")
ax_sp[2].set_xlabel("$\theta\ [rad]$")

#%%----Compute the density distribution----

#Divide the sphere in volume elements, all with the same total volume
r_grid_len = 500
volume = 4/3 * np.pi * a**3 / r_grid_len

#Compute the radii of each volume element to form a grid of radii, adding a last
r_grid = np.zeros(r_grid_len)

for i in range(r_grid_len - 1):
    r_grid[i + 1] = (3/4 / np.pi * volume + r_grid[i]**3)**(1/3)
  
if r_grid[-1] < a:
    r_grid = np.append(r_grid, a)
    
#Assign each particle radius to a radius interval
bins_idx = np.digitize(r, r_grid)

#Compute the density in each radius interval and the Poisson error
p_num = np.zeros(len(r_grid) - 1)
p_idx = np.array(np.unique(bins_idx, return_counts=True))

for i, idx in enumerate(p_idx[0] - 1):
    p_num[idx] += p_idx[1, i]

rho = (p_num / volume) * m_i * M_sun * AU**-3
rho_err = (np.sqrt(p_num) / volume) * m_i * M_sun * AU**-3

#Plot the density at each radius corresponding to the middle of a bin
bar_pos = np.diff(r_grid) / 2 + r_grid[:-1]

fig_rho, ax_rho = plt.subplots()
ax_rho.plot(bar_pos, rho, color="darkcyan", label="Density profile")
ax_rho.axhline(density_0, color="crimson", label="Analytical density profile $\\rho_0$")
ax_rho.fill_between(bar_pos, 
                  rho - rho_err, 
                  rho + rho_err, alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
plt.xlabel("$R\ [AU]$")
plt.ylabel("$\\rho\ [g\ cm^{-3}]$")
plt.legend()

#%%----Mean density of the sphere at various radii----

#Particle radii
R = np.sqrt(x_0**2 + y_0**2 + z_0**2)

#Sphere number density and matter density inside a shell at radius R_i
n_density = np.array([np.sum(R <= R_i) / (4/3 * np.pi * R_i**3) for R_i in np.unique(np.sort(R))])
n_density_err = np.array([np.sqrt(np.sum(R <= R_i)) / (4/3 * np.pi * R_i**3) for R_i in np.unique(np.sort(R))])
rho_mean = m_i * M_sun * n_density * AU**-3
rho_mean_err = m_i * M_sun * n_density_err * AU**-3

fig_n, ax_n = plt.subplots()
ax_n.plot(np.unique(np.sort(R)), rho_mean, color="darkcyan", label="Density within a given radius")
ax_n.axhline(density_0, color="crimson", label="Mean density $\\rho_0$")
ax_n.fill_between(np.unique(np.sort(R)), 
                  rho_mean - rho_mean_err, 
                  rho_mean + rho_mean_err, alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
plt.xlabel("$R\ [AU]$")
plt.ylabel("$\\rho\ [g\ cm^{-3}]$")
plt.legend()
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad

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

N = 40000 #Number of particles
M = 30000 #Total mass in solar masses
a = 200 #Total radius in AU
m_i = M / N #Particle mass in solar masses
equil = False #True if the generated distribution will start in equilibrium
generate = True #True to write the generated distribution to file

#Relative path of the generated file
if equil == True:
    path = "Equilibrium/THS_N" + str(N) + "_M" + str(M) + "_b" + str(a)
else:
    path = "THS_N" + str(N) + "_M" + str(M) + "_a" + str(a)

#%%----Compute theoretical timescales----

#Compute the dynamical time and the collapse time in years

density_0 = M * M_sun / (4/3 * np.pi * (a * AU)**3)

t_dyn = np.sqrt(3 * np.pi / (16 * G_p * density_0)) / year

t_coll = t_dyn / np.sqrt(2)

print("Dynamical time: t_dyn =" + f"{t_dyn: .3f}" + " yr =" + f"{t_dyn * t_iu_yr: .3f}" + " IU")
print("Collapse time: t_coll =" + f"{t_coll: .3f}" + " yr =" + f"{t_coll * t_iu_yr: .3f}" + " IU")
print("Initial density = " + f"{density_0: .3e}" + " g cm^-3")

#%%----GenInput function----

#Function to generate a distribution of particles to input in the Treecode

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
    
    #CM position
    CM_x = m_i * np.sum(x_0) / (m_i*N)
    CM_y = m_i * np.sum(y_0) / (m_i*N)
    CM_z = m_i * np.sum(z_0) / (m_i*N)
    
    #Positions of the particles in the CM frame
    x_CM = x_0 - CM_x
    y_CM = y_0 - CM_y
    z_CM = z_0 - CM_z
    
    r_CM = np.sqrt(x_CM**2 + y_CM**2 + z_CM**2)
    
    #Initial velocities
    vx_0 = np.zeros(N)
    vy_0 = np.zeros(N)
    vz_0 = np.zeros(N)
    
    #Write to file
    if to_file:
        file_path = "Input/" + path + ".in"
        
        with open(file_path, "w") as i_file:
            i_file.write(str(N) + "\n3\n0\n")
            
            for i in tqdm(range(N)):
                i_file.write(str(m_i) + "\n") 
                
            for i in tqdm(range(N)):
                i_file.write(str(x_0[i]) + " " + str(y_0[i]) + " " + str(z_0[i]) + "\n")
            
            for i in tqdm(range(N)):    
                i_file.write(str(vx_0[i]) + " " + str(vy_0[i]) + " " + str(vz_0[i]) + "\n")
        
    return x_0, y_0, z_0, vx_0, vy_0, vz_0, r, phi, theta, x_CM, y_CM, z_CM, r_CM
    
#%%----Generate input----

x_0, y_0, z_0, vx_0, vy_0, vz_0, r, phi, theta, x_CM, y_CM, z_CM, r_CM = GenInput(N, m_i, a, generate)

#%%%Softening parameter and command to run the Treecode

#Softening parameter epsilon
vol = 4/3 * np.pi * np.max(r_CM)**3
eps = 1e-4 * (vol / N)**(1/3)

#Print command to run the Treecode
print('./treecode in="../Truncated Homogeneous Sphere/Input/' + path + '.in"' + 
      ' out="../Truncated Homogeneous Sphere/Output/' + path + '.out"' +
      ' dtime=0.1 eps=' + str(eps) + ' theta=0.1 tstop=30 dtout=0.1' + 
      ' > "../Truncated Homogeneous Sphere/Logs/' + path + '_log"')

#%%----Visualize the generated distribution----

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.set_aspect("equal")
ax.scatter(x_0, y_0, z_0, alpha=0.8, color="darkcyan")

#%%----Plot the spherical coordinates distributions----

fig_sp, ax_sp = plt.subplots(1, 3, figsize=(10, 5))

ax_sp[0].hist(np.log10(r), bins="fd", color="darkcyan")
ax_sp[0].set_xlabel("$\log{r}\ [AU]$")

ax_sp[1].hist(phi, bins="fd", color="darkcyan")
ax_sp[1].set_xlabel("$\phi\ [rad]$")

ax_sp[2].hist(theta, bins="fd", color="darkcyan")
ax_sp[2].set_xlabel("$\\theta\ [rad]$")

#%%----Compute the density distribution----

#Function to divide the distribution in intervals
def DivideDistribution(edge, r_grid_len, equal_volume=False):   
    if equal_volume:
        #Divide the sphere in volume elements, all with the same total volume
        volume = 4/3 * np.pi * edge**3 / r_grid_len

        #Compute the radii of each volume element to form a grid of radii, up to an edge
        r_grid = np.zeros(r_grid_len)

        for i in range(r_grid_len - 1):
            r_grid[i + 1] = (3/4 / np.pi * volume + r_grid[i]**3)**(1/3)
            
        if r_grid[-1] < edge:
            r_grid = np.append(r_grid, edge)
            
    else:
        #Divide the sphere in volume elements, with constant radius intervals
        r_grid = np.linspace(0, edge, r_grid_len)
            
        if r_grid[-1] < edge:
            r_grid = np.append(r_grid, edge)
            
        volume = 4/3 * np.pi * np.array([r_grid[i + 1]**3 - r_grid[i]**3 for i in range(len(r_grid) - 1)])

    return r_grid, volume


#Divide the distribution in radius intervals
r_grid, volume = DivideDistribution(2 * a, 250, False)

#Compute the number of particles inside a given radius interval    
p_num = np.zeros(len(r_grid) - 1)

#Count the number of particles inside every radius interval at a given time
p_idx, p_inside = np.unique(np.digitize(r, r_grid), return_counts=True)

#Count only the particles within a, and add the number to the p_num
#elements at the index of the corresponding occupied interval, leaving
#the unoccupied ones at zero particles    
cond_within = np.where(p_idx < len(r_grid) - 1)
p_num[p_idx[cond_within]] += p_inside[cond_within]

#Compute the density in each radius interval and the Poisson error
rho = (p_num / volume) * m_i * M_sun * AU**-3
rho_err = (np.sqrt(p_num) / volume) * m_i * M_sun * AU**-3

#%%%Plot the density at each radius corresponding to the middle of a bin

#Center of every interval
bar_pos = np.diff(r_grid) / 2 + r_grid[:-1]

fig_rho, ax_rho = plt.subplots()
ax_rho.plot(bar_pos, rho, color="darkcyan", label="Simulation density profile")
ax_rho.axhline(density_0, color="crimson", label="Analytical density profile")
ax_rho.axvline(a, alpha=0.6, color="grey", linestyle="--", label="$r = a$")
ax_rho.fill_between(bar_pos, 
                  rho - rho_err, 
                  rho + rho_err, alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
ax_rho.set_xlabel("$R\ [AU]$")
ax_rho.set_ylabel("$\\rho\ [g\ cm^{-3}]$")
# ax_rho.set_ylim(0, 1.2 * 3 * M / (4 * np.pi* a**3) * M_sun * AU**-3)
ax_rho.legend(loc="upper center")

#%%----Mean density of the sphere at various radii----

#Sphere number density and matter density inside a shell at radius R_i
n_density = np.array([np.sum(r <= R_i) / (4/3 * np.pi * R_i**3) for R_i in np.unique(np.sort(r))])
n_density_err = np.array([np.sqrt(np.sum(r <= R_i)) / (4/3 * np.pi * R_i**3) for R_i in np.unique(np.sort(r))])
rho_mean = m_i * M_sun * n_density * AU**-3
rho_mean_err = m_i * M_sun * n_density_err * AU**-3

#plot the mean density
fig_n, ax_n = plt.subplots()
ax_n.plot(np.unique(np.sort(r)), rho_mean, color="darkcyan", label="Mean density within a given radius")
ax_n.axhline(density_0, color="crimson", label="Analytical density profile")
ax_n.fill_between(np.unique(np.sort(r)), 
                  rho_mean - rho_mean_err, 
                  rho_mean + rho_mean_err, alpha=0.3, color="darkcyan", label="$1\sigma$ Poisson error")
plt.xlabel("$R\ [AU]$")
plt.ylabel("$\\bar{\\rho}\ [g\ cm^{-3}]$")
plt.yscale("log")
plt.legend()

#%%----Plot the potential profile----

V = np.zeros(len(r_grid))

for rad in range(len(r_grid)):
    V[rad] = -G_p * m_i * M_sun * AU**-1 * np.sum(np.sqrt((x_0 - r_grid[rad])**2 + 
                                                    y_0**2 + z_0**2)**-1)
        
#%%%Potential profile plot

#Theoretical profile
@np.vectorize
def V_TH(r):
    if r > a:
        return -G_p * M * M_sun * AU**-1 * r**-1
    else:
        return -G_p * AU**2* 2*np.pi * density_0 * (a**2 - 1/3 * r**2)
        
plt.figure()
plt.plot(r_grid, V, color="darkcyan", label="Simulation potential profile")
plt.plot(r_grid, V_TH(r_grid), color="crimson", label="Theoretical potential profile")
plt.axvline(a, alpha=0.6, color="grey", linestyle="--", label="$r = a$")
plt.xlabel("$r\ [AU]$")
plt.ylabel("$\Phi (r)\ [cm^2 \cdot s^{-2}]$")
plt.legend()

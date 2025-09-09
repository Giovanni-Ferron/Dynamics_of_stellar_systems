# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:57:06 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#%%------------------------------Compute initial conditions------------------------------

G_p = 6.67259 * 10**-8 #G in cgs
M_sun = 1.9891 * 10**33 #solar mass in g
R_sun = 6.9598 * 10**10 #solar radius in cm 
M_earth = 5.976 * 10**27 #earth mass in g
R_earth = 6.378 * 10**8 #earth radius in cm
ly = 9.463 * 10**17 #light year in cm
parsec = 3.086 * 10**18 #parsec in cm
AU = 1.496 * 10**13 #astronomical unit in cm
N_orbits = 2

def v_IU(v_p, M_p=M_sun, r_p=AU):
    return np.sqrt(r_p / (G_p * M_p)) * v_p

def t_IU(t_p, M_p=M_sun, r_p=AU):
    return t_p / (np.sqrt(r_p / (G_p * M_p)) * r_p)

#%%

#%%----------------------------------Read output file------------------------------------

def GetData(filename, N_orbits):
    output = open(filename, "r").read().splitlines()
    data = []
    time = []
    
    for i in range(2, len(output), N_orbits + 2):
        for j in range(0, N_orbits):
            data.append(output[i + j].split())
        
        time.append(float(output[i - 1]))
    
    return np.array(data), np.array(time)
    
data, t = GetData("Output/output_hyp1.out", N_orbits)

#%%

#%%---------------------------Compute coordinates in CM frame----------------------------

#Time in years
t /= 6.283696

#Masses of the objects
m_1 = float(data[0][0])
m_2 = float(data[1][0])
M = m_1 + m_2
mu = (m_1 * m_2) / M

#Positions and velocities in external frame

x_1, y_1, z_1, x_2, y_2, z_2 = [], [], [], [], [], []
vx_1, vy_1, vz_1, vx_2, vy_2, vz_2 = [], [], [], [], [], []

for i in range(0, len(data), N_orbits):
    x_1.append(float(data[i][1]))
    y_1.append(float(data[i][2]))
    # z_1.append(float(data[i][3]))
    
    vx_1.append(float(data[i][4]))
    vy_1.append(float(data[i][5]))
    # vz_1.append(float(data[i][6]))

    x_2.append(float(data[i + 1][1]))
    y_2.append(float(data[i + 1][2]))
    # z_2.append(float(data[i + 1][3]))
    
    vx_2.append(float(data[i + 1][4]))
    vy_2.append(float(data[i + 1][5]))
    # vz_2.append(float(data[i + 1][6]))
    
x_1 = np.array(x_1)
y_1 = np.array(y_1)
# z_1 = np.array(z_1)
x_2 = np.array(x_2)
y_2 = np.array(y_2)
# z_2 = np.array(z_2)

vx_1 = np.array(vx_1)
vy_1 = np.array(vy_1)
# vz_1 = np.array(vz_1)
vx_2 = np.array(vx_2)
vy_2 = np.array(vy_2)
# vz_2 = np.array(vz_2)
    
#Positions and velocities of the CM

x_CM = (m_1 * x_1 + m_2 * x_2) / M
y_CM = (m_1 * y_1 + m_2 * y_2) / M

vx_CM = (m_1 * vx_1 + m_2 * vx_2) / M
vy_CM = (m_1 * vy_1 + m_2 * vy_2) / M

#Positions and velocities in the CM frame

x_cm_1 = x_1 - x_CM
y_cm_1 = y_1 - y_CM

vx_cm_1 = vx_1 - vx_CM
vy_cm_1 = vy_1 - vy_CM

x_cm_2 = x_2 - x_CM
y_cm_2 = y_2 - y_CM

vx_cm_2 = vx_2 - vx_CM
vy_cm_2 = vy_2 - vy_CM

#Position and velocity of the equivalent 1-body problem in polar coordinates

sep_x = x_cm_1 - x_cm_2
sep_y = y_cm_1 - y_cm_2

sep_v_x = vx_cm_1 - vx_cm_2
sep_v_y = vy_cm_1 - vy_cm_2

r = np.sqrt(sep_x**2 + sep_y**2)
theta = np.arctan(sep_y / sep_x)

v_r = (sep_x * sep_v_x + sep_y * sep_v_y) / r
v_theta = r * (sep_v_y / sep_x - sep_y * sep_v_x / sep_x**2) / (1 + (sep_y / sep_x)**2)

#%%

#%%-----------------Compute other relevant quantities in the CM frame--------------------

#Angular momentum per unit mass
l = r * v_theta

#Gravitational potential
V_g = -M / r

#Centripetal potential
V_c = l**2 / (2 * r**2)

#Kinetic energy in the CM frame
E_k = mu * v_r**2 / 2
E_k1 = m_1 * (vx_1**2 + vy_1**2) / 2
E_k2 = m_2 * (vx_2**2 + vy_2**2) / 2

#Total energy in the CM frame
E = E_k + mu * V_c + mu * V_g

#Total energy in the external frame
E_ext = m_1 * (vx_1**2 + vy_1**2) / 2 + m_2 * (vx_2**2 + vy_2**2) / 2 - m_1 * m_2 / (np.sqrt((x_1**2 - x_2**2) + (y_1**2 - y_2**2)))

#Periastron and apoastron
per = abs((-mu * M + np.sqrt( (mu * M)**2 + 2 * mu * l[0]**2 * E[0])) / (2 * E[0]))
apo = abs((-mu * M - np.sqrt( (mu * M)**2 + 2 * mu * l[0]**2 * E[0])) / (2 * E[0]))

#Effective potential
r_range = np.arange(0.05, 10 * apo, 0.01)
V_eff = l[0]**2 / (2 * r_range**2) - M / r_range

#Eccentricity
if E[0] < 0:
    e = (apo - per) / (apo + per)
else:
    e = l[0]**2 / (sep_y[0] * np.sin(-np.arctan(-(sep_y[0] * sep_v_x[0]**2))))
    
#Semi-major axis
if e < 0.999:    
    a = (apo + per) / 2

#Orbital period
if e < 0.999:
    T = 2 * np.pi * np.sqrt(a**3 / M) / 6.283696
    
#Deflection angle
if e > 1:
    # theta_def = 2 * np.arctan(-(sep_y[0] * sep_v_x[0]**2)) - np.pi
    per_idx = np.where(r == min(r))[0][0]
    v_post = (sep_v_x[per_idx:], sep_v_y[per_idx:])
    v_pre = (sep_v_x[per_idx - len(v_post[0]):per_idx], sep_v_y[per_idx - len(v_post[1]):per_idx])
    
    theta_def = np.arccos((v_post[0] * v_pre[0] + v_post[1] * v_pre[1]) / (np.sqrt((v_post[0]**2 + v_post[1]**2) * (v_pre[0]**2 + v_pre[1]**2))))
    theta_def *= (180 / np.pi)
    
#Print the quantities
print("Two body system relevant quantities:")
print("Total energy in the CM frame: E_cm = " + f"{E[0]: .2e}")
print("Total energy in the external frame: E_ext = " + f"{E_ext[0]: .2e}")
print("Angular momentum: l =" + f"{l[0]: .2e}")
print("Eccentricity: e =" + f"{e: .2f}")

if e < 0.999:
    print("Apoastron: APO =" + f"{apo: .2f} AU")
    
print("Periastron: PER =" + f"{per: .2f} AU")

if e < 0.999:
    print("Semi-major axis: a =" + f"{a: .2f} AU")
    print("Orbital period: T =" + f"{T: .2f} yr")
    
if e >= 1:
    print("Deflection angle: theta =" + f"{max(theta_def): .3f}")

#%%

#%%-----------------------------------Plot the orbits------------------------------------

#Plot the effective potential graph

plt.figure(figsize=(20, 20))

plt.plot(r_range, V_eff, lw=5, label="Effective potential")
plt.axhline(E[0] / mu, lw=5, c="red", label="Total energy in the CM frame")
plt.axhline(0., c="black", ls="dotted")
plt.xlabel("x [AU]", size=20)
plt.ylabel("Potential", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlim(0., 8.)
plt.ylim(min(V_eff) - 0.1, 1.0)
plt.grid(color="whitesmoke")
plt.rc('font', **{'size':'20'})
plt.legend(prop={"size": 10})

#Plot the energy balance graph

plt.figure(figsize=(20, 20))

plt.plot(t, E, lw=5, label="Total energy in the CM frame")
plt.plot(t, E_ext, lw=5, label="Total energy in external frame")
plt.plot(t, E_k1, lw=5, label="Kinetic energy of $m_1$")
plt.plot(t, E_k2, lw=5, label="Kinetic energy of $m_2$")
plt.plot(t, mu * V_g, lw=5, label="Gravitational potential energy")
plt.xlabel("t [yr]", size=20)
plt.ylabel("Energy", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.grid(color="whitesmoke")
plt.rc('font', **{'size':'20'})
plt.legend(prop={"size": 10})

#Plot the 1-body orbit

fig, ax = plt.subplots(figsize=(20, 20))

planet = ax.plot(sep_x, sep_y, c="blue", marker="o", markersize=10, label="Mass 1")[0]
orbit = ax.plot(sep_x, sep_y, c="grey", lw=1, zorder=-1)[0]
star = ax.plot(x_CM[0], y_CM[0], c="orange", marker="o", markersize=20, label="Mass 2")
bbox = dict(boxstyle='round', fc='white', ec='black', alpha=0.8)
stats = ax.text(0.875, 0.04, "$m_2 = 1\ M_\odot$\n$m_1 = 10^{-6}\ M_\odot$\n$v_x = $" + f"{sep_v_x[0]: .3f} AU/yr\n$v_y = $" + f"{sep_v_y[0]: .3f} AU/yr", bbox=bbox, color="black", size=10, transform=ax.transAxes)
ax.set_xlabel("x [AU]", size=20)
ax.set_ylabel("y [AU]", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
ax.grid(color="whitesmoke")
ax.legend(prop={"size": 10})

#Animation setup

if e >= 0.999:
    anim_speed = 400
else:
    anim_speed = 1

def update(frame):
    planet.set_xdata([sep_x[anim_speed*frame]])
    planet.set_ydata([sep_y[anim_speed*frame]])
    orbit.set_xdata([sep_x[:anim_speed*frame]])
    orbit.set_ydata([sep_y[:anim_speed*frame]])
    stats.set_text("$m_2 = 1\ M_\odot$\n$m_1 = 10^{-6}\ M_\odot$\n$v_x = $" + f"{sep_v_x[anim_speed*frame]: .3f} AU/yr\n$v_y = $" + f"{sep_v_y[anim_speed*frame]: .3f} AU/yr")
    
    return (planet, orbit)

if e < 0.999:
    end = max(np.where(t < T)[0])
else:
    end = len(sep_x)
    
anim = anim.FuncAnimation(fig=fig, func=update, frames=int(end/anim_speed), interval=2, repeat=True)
plt.show()

#%%
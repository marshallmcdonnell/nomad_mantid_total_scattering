#!/usr/bin/env python
import numpy as np
from scipy.constants import hbar, Avogadro
from scipy.constants import physical_constants

def calc_placzek(q, mass):
    "Input: Q in Anstrom^-1 and Mass in AMU"
    amu_per_kg = 1. / physical_constants['atomic mass constant'][0] 
    ang_per_m = 1. / physical_constants['Angstrom star'][0]
    hbar_amu_ang2_per_s = hbar * amu_per_kg * ang_per_m * ang_per_m  # J*s -> amu*ang^2/s conversion
    hbar2 = hbar_amu_ang2_per_s * hbar_amu_ang2_per_s
    return ( hbar2 / 2.0 ) * ( q*q / mass / Avogadro)


# Create placzek correction vectorizing function
placzek = np.vectorize(calc_placzek)

# Input mass and the Q vector
mass = 50.9415 # amu of Vanadium
q_vector = np.arange(0.1,50.0,0.02)  

# Output the Placzek correction 
placzek_correction = placzek(q_vector, mass)


import matplotlib.pyplot as plt

plt.plot(q_vector, placzek_correction)
plt.show()

#!/usr/bin/env python
import numpy as np
from scipy.constants import m_n, hbar, Avogadro
from scipy.constants import physical_constants

def calc_placzek_first_moment(q, mass):
    "Input: Q in Anstrom^-1 and Mass in AMU"
    amu_per_kg = 1. / physical_constants['atomic mass constant'][0] 
    ang_per_m = 1. / physical_constants['Angstrom star'][0]
    hbar_amu_ang2_per_s = hbar * amu_per_kg * ang_per_m * ang_per_m  # J*s -> amu*ang^2/s conversion
    hbar2 = hbar_amu_ang2_per_s * hbar_amu_ang2_per_s
    return ( hbar2 / 2.0 ) * ( q*q / mass / Avogadro)

def calc_self_placzek( mass_amu, self_scat, theta, incident_path_length, scattered_path_length, detector='1/v'):
    # constants and conversions
    neutron_mass = m_n / physical_constants['atomic mass unit-kilogram relationship'][0]
    angle_conv = np.pi / 180.

    # variables
    l_0 = incident_path_length
    l_s = scattered_path_length
    l_total = l_0 + l_s
    sin_theta = np.sin(theta * angle_conv)

    if detector == '1/v':
        moment_1 = 2. * ( mass_amu / neutron_mass ) * sin_theta * sin_theta * ( l_0 + 3.0 * l_s ) / l_total

    if detector == 'black':
        moment_1 = 4. * ( mass_amu / neutron_mass ) * sin_theta * sin_theta * ( l_0 + 1.5 * l_s ) / l_total

    return self_scat * (1. - moment_1)

'''
# Create placzek correction vectorizing function
placzek = np.vectorize(calc_placzek_first_moment)
mass = 50.9415 # amu of Vanadium
q_vector = np.arange(0.1,50.0,0.02)  
placzek_correction = placzek(q_vector, mass)
'''

mass = 50.9415
self_scat = 0.40
theta = 120.4
l_0 = 19.5
l_1 = 1.11
print calc_self_placzek( mass, self_scat, theta, l_0, l_1 )

placzek = np.vectorize(calc_self_placzek)
offset = 10.0
thetas = np.arange( theta - offset, theta + offset, 0.1)
placzek_out = placzek(mass, self_scat, thetas, l_0, l_1 )

# Output the Placzek correction 
import matplotlib.pyplot as plt

plt.plot(thetas, placzek_out)
plt.show()

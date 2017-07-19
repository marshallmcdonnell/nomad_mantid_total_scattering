#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_n, physical_constants, Planck, Boltzmann, angstrom, Avogadro

moderators = ['ambient', 'ambient_poisoned', 'cold']
parameters = { 'ambient' : [6324, 786, 1.58, 0.099, 0.67143, 0.06075],
               'ambient_poisoned' :  [1200,786,1.58,0.099,0.67143,0.06075],
               'cold' : [3838,1029,2.97,0.089,1.3287,0.14735] 
             }
def delta( lam, lam_1, lam_2):
    return 1. / (1. + np.exp((lam - lam_1)/lam_2))

def phi_m(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2  ):
    return phi_max*(lam_t**4./lam**5.)*np.exp(-(lam_t/lam)**2.)

def phi_e(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2  ):
    return phi_epi*delta(lam, lam_1, lam_2) / (lam**(1+2*a))

def calc_HowellsFunction(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2):
    args = [phi_max, phi_epi, lam_t, a, lam_1, lam_2]
    return phi_m(lam, *args) + phi_e(lam, *args)
    

'''
def calc_HowellsFunction(lambdas, phi_max, phi_epi, lam_t, a, lam_1, lam_2 ):
    term1 = phi_max * ((lam_t**4.)/lambdas**5.)*np.exp(-(lam_t/lambdas)**2.)
    term2 = (phi_epi/(lambdas**(1.+2.*a)))*(1./(1+np.exp((lambdas-lam_1)/lam_2)))
    return term1 + term2
'''


def placzek_self(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2, angle=None, M=14, T=77, R=0.1):
    neutron_mass = m_n / physical_constants['atomic mass unit-kilogram relationship'][0]
   
    def phi_m_1st_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2  ):
        return (2*(lam_t**2/lam**2.) - 5.)*(phi_m(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2  )/lam)

    def phi_m_2nd_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2  ):
        args = [phi_max, phi_epi, lam_t, a, lam_1, lam_2] 
        return (30. - 26.*(lam_t/lam)**2. + 4.*(lam_t/lam)**4.) * (phi_m(lam,*args )/(lam**2.))

    def phi_e_1st_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2  ):
        return -(1+2*a)/lam*phi_e(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2)
    
    def phi_e_2nd_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2  ):
        return ((1+2*a)*(2+2*a))/(lam*lam)*phi_e(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2)

    def F1(lam, lamd=1.44):
        numerator   = (-lam/lamd)*np.exp(-lam/lamd)
        denominator = 1. - np.exp(-lam/lamd)
        return numerator / denominator

    def F2(lam, lamd=1.44):
        return ((lam/lamd)-2.)*F1(lam,lamd)

    def f1(lam, *args):
        return lam*(phi_m_1st_der(lam,*args)+phi_e_1st_der(lam,*args))/(phi_m(lam,*args)+phi_e(lam,*args))

    def f2(lam, *args):
        return lam**2.*(phi_m_2nd_der(lam,*args)+phi_e_2nd_der(lam,*args))/(phi_m(lam,*args)+phi_e(lam,*args))

    def D(R,lam,*args):
        term1 = 0.5*(1+R)**-2.
        term2 = F2(lam) + (9.*R + 3.)*F1(lam) + R*R*f2(lam,*args) + R*(9.*R+1.)*f1(lam,*args) \
                + 2.*R*F1(lam)*f1(lam,*args) + (1. + 5.*R + 16.*R*R)
        return term1*term2

    def E(lam):
        return (Planck**2./(2.*m_n))*(1./(angstrom*lam)**2.)

    args = [phi_max, phi_epi, lam_t, a, lam_1, lam_2]
    angle_conv = np.pi / 180.
    sin_theta      = np.sin(angle * angle_conv)
    sin_theta_by_2 = np.sin(angle * angle_conv / 2.)
    cos_theta = np.cos(angle * angle_conv)
    term1 = 2*(neutron_mass/M)*(1.+R)**-1.
    term2 = (F1(lam) + R*f1(lam,*args) + (2.+3.*R))*sin_theta*sin_theta
    term3 = 0.5*(neutron_mass/(M))*Boltzmann*T/E(lam)
    term4 = (cos_theta + 4.*D(R,lam,*args)*sin_theta_by_2 * sin_theta_by_2) 
    total = 1. - term1*term2 + term3*term4
    return total
    
scale = 1e-2
lines = ['-', ':', '--']
def plot_moderators(x):
    for i, moderator in enumerate(moderators):
        y=scale*calc_HowellsFunction(x,*parameters[moderator])
        plt.plot(x,y,'k'+lines[i])
    plt.legend(moderators,loc='best')
    axes = plt.gca()
    axes.set_ylim([0.,35.])
    axes.set_xlim([0.,3.])
    plt.title('Figure 1: neutron spectrum')
    plt.show()
    return

def plot_placzek_wavelength(x, angle=150., M=14, T=77, R=0.1): 
    for i, moderator in enumerate(moderators):
        args = parameters[moderator] + [angle, M, T, R ] 
        y=placzek_self(x, *args)
        plt.plot(x,y,'k'+lines[i])
    plt.legend(moderators,loc='best')
    #axes = plt.gca()
    #axes.set_ylim([0.70,0.90])
    #axes.set_xlim([0.,3.])
    plt.title('Figure 2: Placzek in wavelength')
    plt.show()
    return

def plot_placzek_momentum_transfer(x, angle=150., M=14, T=77, R=0.1, color='k'):
    angle_conv = np.pi / 180.
    sin_theta = np.sin(angle * angle_conv / 2.)
    for i, moderator in enumerate(moderators):
        args = parameters[moderator] + [angle, M, T, R ] 
        y=placzek_self(x, *args)
        q = (4.*np.pi / x)*sin_theta
        plt.plot(q,y,color+lines[i])
    return

    plot_placzek(q, angle, M, T, R, title=title)
    return

angle=30.
x = np.linspace(0.1,4.0,1000)
plot_moderators(x)
plot_placzek_wavelength(x,angle=150.)

colors = ['k', 'r', 'b', 'g']
for i, angle in enumerate([30., 60., 90., 150.]):
    plot_placzek_momentum_transfer(x,angle=angle,color=colors[i])

labels = list()
for angle in [30., 60., 90., 150.]:
    labels = labels + [ m+' angle: '+str(angle) for m in moderators ]
plt.legend(labels,loc='best')
axes = plt.gca()
axes.set_xlim([0.0,50.])
axes.set_ylim([0.7,1.0])
plt.title('Figure 3: Placzek in Q')
plt.show()



#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_n, physical_constants, Planck, Boltzmann, angstrom, Avogadro

angle_conv = np.pi / 180.


#-------------------------------------------------------------------------------------#
# Moderator function
def delta(lam, lam_1, lam_2):
    return 1. / (1. + np.exp((lam - lam_1) / lam_2))

# Howells function


def phi_m(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2):
    return phi_max * (lam_t**4. / lam**5.) * np.exp(-(lam_t / lam)**2.)


def phi_e(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2):
    return phi_epi * delta(lam, lam_1, lam_2) / (lam**(1 + 2 * a))


def calc_HowellsFunction(lam, *args):
    return phi_m(lam, *args) + phi_e(lam, *args)


# Howells function 1st derivative
def phi_m_1st_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2):
    args = [phi_max, phi_epi, lam_t, a, lam_1, lam_2]
    return (2 * (lam_t / lam)**2. - 5.) * (phi_m(lam, *args) / lam)


def phi_m_2nd_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2):
    args = [phi_max, phi_epi, lam_t, a, lam_1, lam_2]
    term1 = 30. - 26. * (lam_t / lam)**2. + 4. * ((lam_t / lam)**4.)
    term2 = (phi_m(lam, *args) / (lam**2.))
    return term1 * term2


def phi_e_1st_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2):
    return -(1 + 2 * a) / lam * phi_e(lam, phi_max,
                                      phi_epi, lam_t, a, lam_1, lam_2)


def phi_e_2nd_der(lam, phi_max, phi_epi, lam_t, a, lam_1, lam_2):
    numerator = ((1 + 2 * a) * (2 + 2 * a)) * phi_e(lam,
                                                    phi_max, phi_epi, lam_t, a, lam_1, lam_2)
    denominator = (lam * lam)
    return numerator / denominator


def calc_HowellsFunction1stDer(lam, *args):
    return phi_m_1st_der(lam, *args) + phi_e_1st_der(lam, *args)

#-------------------------------------------------------------------------------------#
# Placzek self function (NOTE: needs some of the moderator functions above)


def placzek_self(
        lam,
        phi_max,
        phi_epi,
        lam_t,
        a,
        lam_1,
        lam_2,
        angle=150,
        M=14,
        T=77,
        R=0.1,
        plot_type='full'):
    neutron_mass = m_n / \
        physical_constants['atomic mass unit-kilogram relationship'][0]

    def F1(lam, lamd=1.44):
        numerator = (-lam / lamd) * np.exp(-lam / lamd)
        denominator = 1. - np.exp(-lam / lamd)
        return numerator / denominator

    def F2(lam, lamd=1.44):
        return ((lam / lamd) - 2.) * F1(lam, lamd)

    def f1(lam, *args):
        return lam * (phi_m_1st_der(lam, *args) + phi_e_1st_der(lam,
                                                                *args)) / (phi_m(lam, *args) + phi_e(lam, *args))

    def f2(lam, *args):
        return lam**2. * (phi_m_2nd_der(lam,
                                        *args) + phi_e_2nd_der(lam,
                                                               *args)) / (phi_m(lam,
                                                                                *args) + phi_e(lam,
                                                                                               *args))

    def D(R, lam, *args):
        term1 = (1. + R)**-2.
        term2 = F2(lam)
        term3 = (9. * R + 3.) * F1(lam)
        term4 = R * R * f2(lam, *args)
        term5 = R * (9. * R + 1.) * f1(lam, *args)
        term6 = 2. * R * F1(lam) * f1(lam, *args)
        term7 = (1. + 5. * R + 16. * R * R)
        total = 0.5 * term1 * (term2 + term3 + term4 + term5 + term6 + term7)
        return total

    def E(lam):
        return (Planck**2. / (2. * m_n)) * (1. / (angstrom * lam)**2.)

    args = [phi_max, phi_epi, lam_t, a, lam_1, lam_2]

    sin_theta = np.sin(angle * angle_conv)
    sin_theta_by_2 = np.sin(angle * angle_conv / 2.)
    cos_theta = np.cos(angle * angle_conv)

    term1 = 2 * (neutron_mass / M) * (1. + R)**-1.
    term2 = (F1(lam) + R * f1(lam, *args) + (2. + 3. * R)) * \
        sin_theta_by_2 * sin_theta_by_2
    term3 = 0.5 * (neutron_mass / (M)) * (Boltzmann * T) / E(lam)
    term4 = (cos_theta + (4. * D(R, lam, *args)
                          * sin_theta_by_2 * sin_theta_by_2))

    if plot_type == '1-t2':
        total = 1. - term2
    if plot_type == '1-t1*t2':
        total = 1. - term1 * term2
    if plot_type == '1+t4':
        total = 1. + term3 * term4
    if plot_type == '1+t3*t4':
        total = 1. + term3 * term4
    if plot_type == '1-t1*t2+t3*t4' or plot_type == 'full':
        total = 1. - term1 * term2 + term3 * term4

    return total

#-------------------------------------------------------------------------------------#
# Plots of Howells in 1984


def plot_moderators(x, incident_spectrums, lines=['k-', 'k--', 'k:']):
    for i, key in enumerate(incident_spectrums):
        y = calc_HowellsFunction(x, *incident_spectrums[key])
        plt.plot(x, y, lines[i])
    plt.legend(list(incident_spectrums.keys()), loc='best')
    axes = plt.gca()
    axes.set_ylim([0., 35.])
    axes.set_xlim([0., 3.])
    plt.title('Figure 1: neutron spectrum')
    plt.show()
    return


def plot_moderators_ratio_f_prime_over_f(
    x, incident_spectrums, lines=[
        'k-', 'k--', 'k:']):
    for i, key in enumerate(incident_spectrums):
        f = calc_HowellsFunction(x, *incident_spectrums[key])
        fprime = calc_HowellsFunction1stDer(x, *incident_spectrums[key])
        plt.plot(x, fprime / f, lines[i])
    plt.legend(list(incident_spectrums.keys()), loc='best')
    #axes = plt.gca()
    # axes.set_ylim([0.,35.])
    # axes.set_xlim([0.,3.])
    plt.title("Figure f'/f")
    plt.show()
    return


def plot_placzek_wavelength(
    x,
    incident_spectrums,
    angle=150.,
    M=14,
    T=77,
    R=0.1,
    lines=[
        'k-',
        'k--',
        'k:']):
    for i, key in enumerate(incident_spectrums):
        args = incident_spectrums[key] + [angle, M, T, R]
        y = placzek_self(x, *args)
        plt.plot(x, y, lines[i])
    plt.legend(list(incident_spectrums.keys()), loc='best')
    axes = plt.gca()
    axes.set_ylim([0.75, 0.88])
    axes.set_xlim([0., 3.])
    locs, labs = plt.yticks()
    plt.yticks([0.85, 0.80])

    plt.title('Figure 2: Placzek in wavelength')
    plt.show()
    return


def ConvertLambdaToQ(lam, angle):
    sin_theta_by_2 = np.sin(angle * angle_conv / 2.)
    q = (4. * np.pi / lam) * sin_theta_by_2
    return q


def plot_placzek_momentum_transfer(
    x,
    incident_spectrums,
    angle=150.,
    M=14,
    T=77,
    R=0.1,
    color='k',
    plot_type='full',
    lines=[
        '-',
        '--',
        ':']):
    for i, moderator in enumerate(incident_spectrums):
        args = incident_spectrums[moderator] + [angle, M, T, R]
        y = placzek_self(x, *args, plot_type=plot_type)
        q = ConvertLambdaToQ(x, angle)
        plt.plot(q, y, color + lines[i])
    return

#-------------------------------------------------------------------------------------#
# Main code


if '__main__' == __name__:
    '''
    Reproducing the results of the paper:

    W. S. Howells
    "On the choice of moderator for a liquids diffractometer on a pulsed neutron source."
    Nuclear Instruments and Methods in Physics Research, 223, 1984, pp 141-146
    '''

    # Table 1: Moderator parameters
    # NOTE: have scaled phi_max and phi_epi to get the correct scale (here,
    # have multiplied by 1e-2) in Fig. 1

    incident_spectrums = collections.OrderedDict()
    # moderator_type              phi_max  phi_epi  lam_t  a      lam_1
    # lam_2
    incident_spectrums['ambient'] = [
        63.24, 7.86, 1.58, 0.099, 0.67143, 0.06075]
    incident_spectrums['cold'] = [38.38, 10.29, 2.97, 0.089, 1.32870, 0.14735]
    incident_spectrums['ambient_poisoned'] = [
        12.00, 7.86, 1.58, 0.099, 0.67143, 0.06075]

    # Create wavelength vector for lamda min to lambda max
    lam_lo = 0.1
    lam_hi = 7.4
    x = np.linspace(lam_lo, lam_hi, 1000)

    # Figure 1
    plot_moderators(x, incident_spectrums)
    plot_moderators_ratio_f_prime_over_f(x, incident_spectrums)

    # Figure 2
    plot_placzek_wavelength(x, incident_spectrums)

    # Figure 3
    plot_placzek_momentum_transfer(x, incident_spectrums, angle=150.)
    axes = plt.gca()
    axes.set_xlim([0.0, 60.])
    axes.set_ylim([0.75, .88])
    locs, labs = plt.yticks()
    plt.yticks([0.85, 0.80])
    plt.legend(list(incident_spectrums.keys()))
    plt.title('Figure 3\nNOTE: Cannot produce low-Q behavior of cold moderator \n \
               without comprising ambient moderators')
    plt.show()

    # Drop the ambient poisoned moderator not used in Figure 4
    incident_spectrums.pop('ambient_poisoned', None)

    colors = ['k', 'r', 'b', 'g']
    # for plot_type in ['1-t2', '1-t1*t2', '1+t4', '1+t3*t4', '1-t1*t2+t3*t4']:
    for plot_type in ['1-t1*t2+t3*t4']:
        for i, angle in enumerate([30., 60., 90., 150.]):
            plot_placzek_momentum_transfer(
                x,
                incident_spectrums,
                plot_type=plot_type,
                angle=angle,
                color=colors[i])

        labels = list()
        for angle in [30., 60., 90., 150.]:
            labels = labels + [m + ' angle: ' +
                               str(angle) for m in incident_spectrums]
        plt.legend(labels, loc='best')
        plt.figtext(0.65, 0.86, ' 30 deg', size='large')
        plt.figtext(0.70, 0.80, ' 60 deg', size='large')
        plt.figtext(0.75, 0.70, ' 90 deg', size='large')
        plt.figtext(0.80, 0.50, '150 deg', size='large')
        axes = plt.gca()
        axes.set_xlim([0.0, 50.])
        axes.set_ylim([0.7, 1.0])

        plt.title('Figure 4')

        plt.show()

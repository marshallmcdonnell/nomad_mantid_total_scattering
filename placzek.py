#!/usr/bin/env python
import sys
import json
import glob
import re
import ConfigParser
from h5py import File
from mantid import mtd
from mantid.simpleapi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_n, hbar, Avogadro, micro
from scipy.constants import physical_constants
from scipy import interpolate, signal, ndimage, optimize


#-----------------------------------------------------------------------------------------#
# Functions for fitting the incident spectrum
def getFitRange(x, y, x_lo, x_hi):
    if x_lo is None:
        x_lo = min(x)
    if x_hi is None:
        x_hi = max(x)

    x_fit = x[ (x >= x_lo) & (x <= x_hi)]
    y_fit = y[ (x >= x_lo) & (x <= x_hi)]
    return x_fit, y_fit

def fitCubicSpline(x, y, x_lo=None, x_hi=None,s=1e15):
    x_fit, y_fit = getFitRange(x, y, x_lo, x_hi)
    tck = interpolate.splrep(x_fit,y_fit,s=s)
    fit = interpolate.splev(x,tck,der=0)
    fit_prime = interpolate.splev(x,tck,der=1)
    return fit, fit_prime

def fitHowellsFunction(x, y, x_lo=None, x_hi=None ):
    # Fit with analytical function from HowellsEtAl
    def calc_HowellsFunction(lambdas, phi_max, phi_epi, lam_t, lam_1, lam_2, a ):
        term1 = phi_max * ((lam_t**4.)/lambdas**5.)*np.exp(-(lam_t/lambdas)**2.) 
        term2 = (phi_epi/(lambdas**(1.+2.*a)))*(1./(1+np.exp((lambdas-lam_1)/lam_2)))
        return term1 + term2

    def calc_HowellsFunction1stDerivative(lambdas, phi_max, phi_epi, lam_t, lam_1, lam_2, a ):
        term1 = (((2*lam_t**2)/lambdas**2) - 5.) * (1./lambdas) * phi_max * ((lam_t**4.)/lambdas**5.)*np.exp(-(lam_t/lambdas)**2.) 
        term2 = ((1+2*a)/lambdas)*(1./lambdas)*(phi_epi/(lambdas**(1.+2.*a)))*(1./(1+np.exp((lambdas-lam_1)/lam_2)))
        return term1 + term2

    x_fit, y_fit = getFitRange(x, y, x_lo, x_hi)
    params = [1.,1.,1.,0.,1.,1.]
    params, convergence = optimize.curve_fit( calc_HowellsFunction, x_fit, y_fit, params)
    fit = calc_HowellsFunction(x, *params)
    fit_prime = calc_HowellsFunction1stDerivative(x, *params)
    return fit, fit_prime

def fitCubicSplineWithGaussConv(x, y, x_lo=None, x_hi=None):
    # Fit with Cubic Spline using a Gaussian Convolution to get weights
    def moving_average(y, sigma=3):
        b = signal.gaussian(39, sigma)
        average = ndimage.filters.convolve1d(y, b/b.sum())
        var = ndimage.filters.convolve1d(np.power(y-average,2),b/b.sum())
        return average, var

    x_fit, y_fit = getFitRange(x, y, x_lo, x_hi)
    avg, var = moving_average(y_fit)
    print x_fit
    spline_fit = interpolate.UnivariateSpline(x_fit, y_fit, w=1./np.sqrt(var))
    spline_fit_prime = spline_fit.derivative()
    return spline_fit, spline_fit_prime 


#-----------------------------------------------------------------------------------------#
#Get incident spectrum from Monitor 

def plotIncidentSpectrum(x, y, fit, fit_prime, title=None):
    plt.plot(x,y,'bo',x,fit,'--')
    plt.legend(['Incident Spectrum','Fit f(x)'],loc='best')
    if title is not None:
        plt.title(title)
    plt.show()

    plt.plot(x,fit_prime/fit,'x--',label="Fit f'(x)/f(x)")
    plt.xlabel('Wavelength')
    plt.legend()
    if title is not None:
        plt.title(title)
    axes = plt.gca()
    axes.set_ylim([-12,6])
    plt.show()
    return

def getIncidentSpectrumFromMonitor(van_scans, incident=0, transmission=1, lam_binning="0.1,0.02,3.1"):
    van = ','.join(['NOM_%d' % num for num in van_scans])

    #-------------------------------------------------
    # Joerg's read_bm.pro code
    p = 0.000794807
    
    # get delta lambda from lamda binning
    lam_bin = float(lam_binning.split(',')[1])


    #-------------------------------------------------
    # Version 2: Use conversions in mantid
    monitor_raw = LoadNexusMonitors(van)
    monitor = 'monitor'
    NormaliseByCurrent(InputWorkspace=monitor_raw, OutputWorkspace=monitor,   
                       RecalculatePCharge=True)
    ConvertUnits(InputWorkspace=monitor, OutputWorkspace=monitor,
                 Target='Wavelength', EMode='Elastic')
    monitor = Rebin(InputWorkspace=monitor, Params=lam_binning, PreserveEvents=True)

    lam = monitor.readX(incident)[1:]
    bm  = monitor.readY(incident)
    e0 = 5333.0 * lam / 1.8 * 2.43e-5 * p
    bmeff = bm / ( 1. - np.exp(-e0*.1))
    #bmeff = bm / ( 1. - np.exp((-1./1.44)*lam))
    bmpp = bmeff / lam_bin 
   
    total_intensity = 0.
    for d_lam, d_bm in zip(lam,bmpp):
        if d_lam >= 0.1 and d_lam <= 2.9:
            total_intensity += d_lam*d_bm
    print "Version 2 Total intensity:", total_intensity / 1.e8, "10^8 neutrons per s"
    '''
    plt.plot(lam, bmpp, 'x-')
    xunit = monitor.getAxis(0).getUnit()
    plt.xlabel(str(xunit.caption())+' ('+str(xunit.symbol())+')')
    yunit = monitor.getAxis(1).getUnit()
    plt.ylabel(str(yunit.caption())+' ('+str(yunit.symbol())+')')
    plt.show()
    '''
    incident_ws = CreateWorkspace(DataX=lam, DataY=bmeff, UnitX='Wavelength')
    return incident_ws

def calc_self_placzek( incident_ws, atom_species, theta, L1, L2, detector_alpha=None, detector_lambda_d = 1.44,
                       FitSpectrum='GaussConvCubicSpline', Binning="0.15,0.05,3.2", detector='1/v'):
    # constants and conversions
    neutron_mass = m_n / physical_constants['atomic mass unit-kilogram relationship'][0]

    # variables
    L_total = L1 + L2
    f = L1 / L_total

    angle_conv = np.pi / 180.
    sin_theta = np.sin(theta * angle_conv)
    sin_theta_by_2 = np.sin(theta * angle_conv / 2.)

    # get derivative of incident spectrum using cubic spline
    incident_monitor = 0
    x_lambda = incident_ws.readX(incident_monitor)
    y_intensity = incident_ws.readY(incident_monitor)

    # Fit Incident Spectrum to get phi_1(lambda)
    lam_lo = float(Binning.split(',')[0])
    lam_hi = float(Binning.split(',')[2])
    
    if FitSpectrum == 'CubicSpline': 
        fit, fit_prime = fitCubicSpline(x_lambda, y_intensity, x_lo=lam_lo, x_hi=lam_hi,s=1e7)
        plotIncidentSpectrum(x_lambda, y_intensity, fit, fit_prime, title='Simple Cubic Spline: Default')

    if FitSpectrum == 'HowellsFunction': 
        fit, fit_prime = fitHowellsFunction(x_lambda, y_intensity, x_lo=lam_lo, x_hi=lam_hi)
        plotIncidentSpectrum(x_lambda, y_intensity, fit, fit_prime, title='HowellsFunction')

    if FitSpectrum == 'GaussConvCubicSpline':
        print x_lambda
        spline_fit, spline_fit_prime =  fitCubicSplineWithGaussConv(x_lambda, y_intensity,x_lo=lam_lo,x_hi=lam_hi)
        fit = spline_fit(x_lambda)
        fit_prime = spline_fit_prime(x_lambda)
        plotIncidentSpectrum(x_lambda, y_intensity, fit, fit_prime, title='Cubic Spline w/ Gaussian Kernel Convolution ')

    phi_1 = x_lambda * fit_prime / fit
    
    # Set detector exponential coefficient alpha
    if detector_alpha is None:
        detector_alpha = 2.* np.pi / detector_lambda_d

    # Detector law to get eps_1(lambda) 
    if detector == '1/v':
        c = -detector_alpha / (2. * np.pi)
        x = x_lambda
        detector_law_term = c*x*np.exp(c*x) / (1. - np.exp(c*x))

    eps_1 = detector_law_term

    # Placzek
    term1 = (f - 1.) * phi_1
    term2 = f*eps_1
    term3 = f - 3.
    
    term4 = 0.0
    for species in atom_species:
        term4 += species['concentration'] * species['sqrdScatLengthBar'] * neutron_mass / species['amu']
    
    placzek_correction = 2.*( term1 - term2 + term3) * sin_theta_by_2 * sin_theta_by_2 * term4
    return placzek_correction


#-----------------------------------------------------------------------------------------#
# Start Placzek calculations

if '__main__' == __name__:
    # Get input parameters
    configfile = sys.argv[1]
    with open(configfile) as handle:
        config = json.loads(handle.read())

    van_scans = config['van']
    van_bg = config['van_bg']
    van_corr_type = config.get('van_corr_type', "Carpenter")
    calib = str(config['calib'])
    charac = str(config['charac'])
    binning= config['binning']
    cache_dir = str(config.get("CacheDir", os.path.abspath('.') ))

    van = ','.join(['NOM_%d' % num for num in van_scans])
    van_bg = ','.join(['NOM_%d' % num for num in van_bg])

    results = PDLoadCharacterizations(Filename=charac, OutputWorkspace='characterizations')
    alignAndFocusArgs = dict(PrimaryFlightPath = results[2],
                             SpectrumIDs       = results[3],
                             L2                = results[4],
                             Polar             = results[5],
                             Azimuthal         = results[6])

    alignAndFocusArgs['CalFilename'] = calib
    #alignAndFocusArgs['GroupFilename'] don't use
    #alignAndFocusArgs['Params'] use resampleX
    alignAndFocusArgs['ResampleX'] = -6000
    alignAndFocusArgs['Dspacing'] = True
    #alignAndFocusArgs['PreserveEvents'] = True 
    alignAndFocusArgs['RemovePromptPulseWidth'] = 50
    alignAndFocusArgs['MaxChunkSize'] = 8
    #alignAndFocusArgs['CompressTolerance'] use defaults
    #alignAndFocusArgs['UnwrapRef'] POWGEN option
    #alignAndFocusArgs['LowResRef'] POWGEN option
    #alignAndFocusArgs['LowResSpectrumOffset'] POWGEN option
    #alignAndFocusArgs['CropWavelengthMin'] from characterizations file
    #alignAndFocusArgs['CropWavelengthMax'] from characterizations file
    alignAndFocusArgs['Characterizations'] = 'characterizations'
    alignAndFocusArgs['ReductionProperties'] = '__snspowderreduction'
    alignAndFocusArgs['CacheDir'] = cache_dir



    for binsize in [0.05]:
        lam_lo = 0.1
        lam_hi = 3.1
        lam_binning = ','.join([ str(x) for x in [lam_lo,binsize,lam_hi]])
        print "Wavelength Binning:", lam_binning
        incident_ws = getIncidentSpectrumFromMonitor(van_scans,lam_binning=lam_binning)
        SetSampleMaterial(incident_ws, ChemicalFormula='Si')
        mass = incident_ws.sample().getMaterial().relativeMolecularMass()

        

        # Fitting using a cubic spline 
        theta = 120.4
        l_0 = 19.5
        l_1 = 1.11
        placzek_out =  calc_self_placzek( incident_ws, mass, theta, l_0, l_1, Binning=lam_binning )

        # Output the Placzek correction 
        import matplotlib.pyplot as plt
        lam = incident_spectrum.readX(incident)[1:]
        plt.plot(lam, placzek_out)
        plt.show()


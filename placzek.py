#!/usr/bin/env python
import sys
import json
import collections
from mantid import mtd
from mantid.simpleapi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_n, physical_constants
from scipy import interpolate, signal, ndimage, optimize

def ConvertLambdaToQ(lam,angle):
    angle_conv = np.pi / 180.
    sin_theta_by_2 = np.sin(angle * angle_conv / 2.)
    q = (4.*np.pi / lam)*sin_theta_by_2
    return q

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

def fitCubicSplineViaMantidSplineSmoothing(InputWorkspace, **kwargs):
    SplineSmoothing(InputWorkspace, OutputWorkspace='fit', OutputWorkspaceDeriv='fit_prime', DerivOrder=1,**kwargs)
    return mtd['fit'].readY(0), mtd['fit_prime_1'].readY(0) 

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

def fitCubicSplineWithGaussConv(x, y, x_lo=None, x_hi=None, sigma=3):
    # Fit with Cubic Spline using a Gaussian Convolution to get weights
    def moving_average(y, sigma=sigma):
        b = signal.gaussian(39, sigma)
        average = ndimage.filters.convolve1d(y, b/b.sum())
        var = ndimage.filters.convolve1d(np.power(y-average,2),b/b.sum())
        return average, var

    x_fit, y_fit = getFitRange(x, y, x_lo, x_hi)
    avg, var = moving_average(y_fit)
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

def GetIncidentSpectrumFromMonitor(Filename, OutputWorkspace=None, incident=0, transmission=1, lam_binning="0.1,0.02,3.1"):
    scans = ','.join(['NOM_%d' % num for num in Filename])

    #-------------------------------------------------
    # Joerg's read_bm.pro code
    p = 0.000794807
    
    # get delta lambda from lamda binning
    lam_bin = float(lam_binning.split(',')[1])


    #-------------------------------------------------
    # Version 2: Use conversions in mantid
    monitor_raw = LoadNexusMonitors(scans)
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
    incident_ws = CreateWorkspace(DataX=lam, DataY=bmeff, OutputWorkspace=OutputWorkspace, UnitX='Wavelength')
    return incident_ws


def FitIncidentSpectrum(InputWorkspace, OutputWorkspace,FitSpectrumWith='GaussConvCubicSpline', Binning="0.15,0.05,3.2"):

    incident_ws = mtd[InputWorkspace]

    # Fit Incident Spectrum  
    incident_index = 0
    x_lambda = incident_ws.readX(incident_index)
    y_intensity = incident_ws.readY(incident_index)

    lam_lo = float(Binning.split(',')[0])
    lam_hi = float(Binning.split(',')[2])
    
    if FitSpectrumWith == 'CubicSpline': 
        fit, fit_prime = fitCubicSpline(x_lambda, y_intensity, x_lo=lam_lo, x_hi=lam_hi,s=1e7)
        plotIncidentSpectrum(x_lambda, y_intensity, fit, fit_prime, title='Simple Cubic Spline: Default')

    elif FitSpectrumWith == 'CubicSplineViaMantid': 
        fit, fit_prime = fitCubicSplineViaMantidSplineSmoothing(InputWorkspace=InputWorkspace,MaxNumberOfBreaks=8)
        plotIncidentSpectrum(x_lambda, y_intensity, fit, fit_prime, title='Cubic Spline via Mantid SplineSmoothing')
    elif FitSpectrumWith == 'HowellsFunction': 
        fit, fit_prime = fitHowellsFunction(x_lambda, y_intensity, x_lo=lam_lo, x_hi=lam_hi)
        plotIncidentSpectrum(x_lambda, y_intensity, fit, fit_prime, title='HowellsFunction')

    elif FitSpectrumWith == 'GaussConvCubicSpline':
        spline_fit, spline_fit_prime =  fitCubicSplineWithGaussConv(x_lambda, y_intensity,x_lo=lam_lo,x_hi=lam_hi,sigma=2)
        fit = spline_fit(x_lambda)
        fit_prime = spline_fit_prime(x_lambda)
        plotIncidentSpectrum(x_lambda, y_intensity, fit, fit_prime, title='Cubic Spline w/ Gaussian Kernel Convolution ')

    else:
        raise Exception("Unknown method for fitting incident spectrum")
        return

    CreateWorkspace(DataX=x_lambda, DataY=np.append(fit,fit_prime), OutputWorkspace=OutputWorkspace, UnitX='Wavelength', NSpec=2)
    return mtd[OutputWorkspace]


def CalculatePlaczekSelf( incident_ws, atom_species, theta, L1, L2,
                          detector_alpha=None, detector_lambda_d = 1.44,
                           detector='1/v'):
    # constants and conversions
    neutron_mass = m_n / physical_constants['atomic mass unit-kilogram relationship'][0]

    # variables
    L_total = L1 + L2
    f = L1 / L_total

    angle_conv = np.pi / 180.
    sin_theta = np.sin(theta * angle_conv)
    sin_theta_by_2 = np.sin(theta * angle_conv / 2.)

    # get derivative of incident spectrum using cubic spline
    incident_index = 0
    incident_prime_index = 1

    x_lambda = mtd[incident_ws].readX(incident_index)
    incident = mtd[incident_ws].readY(incident_index)
    incident_prime = mtd[incident_ws].readY(incident_prime_index)

    phi_1 = x_lambda * incident_prime / incident 
    
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
    for species, props in atom_species.iteritems():
        term4 += props['concentration'] * props['tot_scatt_length'] * neutron_mass / props['mass']
    
    placzek_correction = 2.*( term1 - term2 + term3) * sin_theta_by_2 * sin_theta_by_2 * term4
    return placzek_correction


#-----------------------------------------------------------------------------------------#
# Start Placzek calculations

if '__main__' == __name__:
    # Get input parameters
    configfile = sys.argv[1]
    with open(configfile) as handle:
        config = json.loads(handle.read())

    # Get sample info
    sample = config['sam']
    can = sample['Background']

    # Get normalization info
    van = config['van']
    calib = str(config['calib'])
    charac = str(config['charac'])
    binning= config['binning']
    cache_dir = str(config.get("CacheDir", os.path.abspath('.') ))

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


    # Get incident spectrum
    print "Processing Scan: ", sample['Runs'] 

    incident_ws = 'incident_ws'
    lam_binning = str(sample['InelasticCorrection']['LambdaBinning'])
    GetIncidentSpectrumFromMonitor(sample['Runs'], OutputWorkspace=incident_ws, lam_binning=lam_binning)


        
    # Fit incident spectrum
    incident_fit = 'incident_fit'
    fit_type = str(sample['InelasticCorrection']['FitSpectrumWith'])
    FitIncidentSpectrum(InputWorkspace=incident_ws, OutputWorkspace=incident_fit, FitSpectrumWith=fit_type, Binning=lam_binning)

    # Set sample info
    SetSampleMaterial(incident_fit, ChemicalFormula=str(sample['Material']))

    # Get concentrations from formula
    #       Note: for chemicalFormula 
    #                 index == 0 is atom info, 
    #                 index == 1 is stoichiometry
    total_stoich = 0.0
    material =  mtd[incident_fit].sample().getMaterial().chemicalFormula()
    atom_species = collections.OrderedDict()
    for atom, stoich in zip(material[0], material[1]):
        totalScattLength = atom.neutron()['tot_scatt_length'] / 10.
        atom_species[atom.symbol] = {'mass' : atom.mass,
                                    'stoich' : stoich,
                                    'tot_scatt_length' : totalScattLength }
        total_stoich += stoich

    for atom, props in atom_species.iteritems():
        props['concentration'] = props['stoich'] / total_stoich        
    
    # Parameters for NOMAD detectors by bank
    L1 = 19.5
    banks = collections.OrderedDict()
    banks[6] = { 'L2'    :   2.06, 'theta' :   8.60 }
    banks[1] = { 'L2'    :   2.01, 'theta' :  15.10 }
    banks[2] = { 'L2'    :   1.68, 'theta' :  31.00 }
    banks[3] = { 'L2'    :   1.14, 'theta' :  65.00 }
    banks[4] = { 'L2'    :   1.11, 'theta' : 120.40 }
    banks[5] = { 'L2'    :   0.79, 'theta' : 150.10 }


    # Apply bank-by-bank Placzek correction   
    import matplotlib.pyplot as plt
    bank_colors = ['k', 'r', 'b', 'g', 'y', 'c'] 
    for i, (bank, params) in enumerate(banks.iteritems()):
        L2 = params['L2']
        theta = params['theta']
        
        # Calculate 1st order self-scattering Placzek correction 
        placzek_out =  CalculatePlaczekSelf( incident_fit, atom_species, theta, L1, L2 )

        # Output the Placzek correction 
        incident = 0
        lam = mtd[incident_fit].readX(incident)
        q = ConvertLambdaToQ(lam,theta)

        label= 'Bank: %d at Theta %d' % (bank,int(theta))
        plt.plot(q,   placzek_out, bank_colors[i]+'-', label=label)

    material = ' '.join([symbol+str(int(props['stoich']))+' ' for symbol, props in atom_species.iteritems()])
    plt.title('Placzek vs. Q for '+material)
    plt.xlabel('Q (Angstroms^-1')
    plt.ylabel('P(Q)')
    plt.legend()
    plt.show()

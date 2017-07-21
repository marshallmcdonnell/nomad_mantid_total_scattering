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

#-----------------------------------------------------------------------------------
# . NexusHandler

def findacipts(NOMhome):
    # find all accessible IPTS
    return alliptsnr

def parseInt(number):
    try:
        return int(number)
    except ValueError, e:
        raise Exception("Invalid scan numbers: %s" % str(e))

    return 0

def procNumbers(numberList):
    numberList = [ num for num in str(','.join(numberList)).split(',') ]

    result = []
    if isinstance(numberList,str):
        if "-" in numberList:
            item = [parseInt(i) for i in numberList.split("-")]
            if item[0] is not None:
                result.extend(range(item[0], item[1]+1))

    else:
        for item in numberList:
            # if there is a dash then it is a range
            if "-" in item:
                item = [parseInt(i) for i in item.split("-")]
                item.sort()
                if item[0] is not None:
                    result.extend(range(item[0], item[1]+1))
            else:
                item = parseInt(item)
                if item:
                    result.append(item)

    result.sort()
    return result

class NexusHandler(object):
    def __init__(self, instrument='NOM', cfg_filename='nomad_config.cfg'):
        self._makeScanDict()

        config_path = '/SNS/' + instrument + '/shared/' + cfg_filename
        config = ConfigParser.SafeConfigParser()
        config.read(config_path)
        self._props = { name : path for name, path in config.items('meta') }
        self._props.update({ name : path for name, path in config.items('nexus') })

    def listProps(self):
        return self._props.keys()

    def getNxData(self,scans,props):
        scans = [ str(scan) for scan in scans ]
        scans = procNumbers(scans)
        
        scansInfo = dict()
        for scan in scans:
            scanInfo=self._scanDict[str(scan)]
            nf=File(scanInfo['path'],'r')
            prop_dict = { prop : self._props[prop] for prop in props }
            for key, path in prop_dict.iteritems():
                try:
                    scanInfo.update( { key : nf[path][0] } )
                except KeyError:
                    pass
            scansInfo.update(scanInfo)
        return scansInfo


    def _makeScanDict(self, facility='SNS', instrument='NOM'):
        scanDict = {}
        instrument_path = '/'+facility+'/'+instrument+'/'
        for ipts in os.listdir(instrument_path):
            if ipts.startswith('IPTS'):
                num = ipts.split('-')[1]
                ipts_path = instrument_path+ipts+'/'
                if os.path.isdir(ipts_path+'nexus'):
                    for scanpath in sorted(glob.glob(ipts_path+'nexus/NOM_*')):
                        scan = str(re.search(r'NOM_(\d+)\.nxs', scanpath).group(1))
                        scanDict[scan] = {'ipts' : num, 'path' : scanpath, 'format' : 'nexus' }

                elif os.path.isdir(ipts_path+'0'):
                    for scanDir in glob.glob(ipts_path+'0/*'):
                        scan = str(re.search(r'\/0\/(\d+)', scanDir).group(1))
                        scanpath = scanDir+'/NeXus/NOM_'+scan+'_event.nxs'
                        scanDict[scan] = {'ipts' : num, 'path' : scanpath, 'format' : 'prenexus' }
        self._scanDict = scanDict

#-----------------------------------------------------------------------------------------
# Absolute Scale stuff
def combine_dictionaries( dic1, dic2 ):
    result = dict()
    for key in (dic1.viewkeys() | dic2.keys()):
        print key, dic1[key]
        if key in dic1: result.setdefault(key, {}).update(dic1[key])
        if key in dic2: result.setdefault(key, {}).update(dic2[key])
    return result

class Shape(object):
    def __init__(self):
        self.shape = None
    def getShape(self):
        return self.shape

class Cylinder(Shape):
    def __init__(self):
        self.shape = 'Cylinder'
    def volume(self, Radius=None,Height=None):
        return np.pi * Height * Radius * Radius

class Sphere(Shape):
    def __init__(self):
        self.shape = 'Sphere'
    def volume(self, Radius=None):
        return (4./3.) * np.pi * Radius * Radius * Radius

class GeometryFactory(object):

    @staticmethod
    def factory(Geometry):
        factory = { "Cylinder" : Cylinder(),
                    "Sphere"   : Sphere() }
        return factory[Geometry["Shape"]]


nf = NexusHandler()
def getAbsScaleInfoFromNexus(scans,ChemicalFormula=None,Geometry=None,PackingFraction=None,BeamWidth=1.8,SampleMassDensity=None):
    # get necessary properties from Nexus file
    props = ["formula", "mass", "mass_density", "sample_diameter", "sample_height", "items_id"]
    info = nf.getNxData(scans,props)
    info['sample_diameter'] =  0.1 * info['sample_diameter'] # mm -> cm

    for key in info:
        print key, info[key]

    if ChemicalFormula:
        info["formula"] = ChemicalFormula
    if SampleMassDensity:
        info["mass_density"] = SampleMassDensity

    # setup the geometry of the sample
    if Geometry is None:
        Geometry = dict()
    if "Shape" not in Geometry:
        Geometry["Shape"] = 'Cylinder'
    if "Radius" not in Geometry:
         Geometry['Radius'] = info['sample_diameter']/2.
    if "Height" not in Geometry:
         Geometry['Height'] = info['sample_height']

    if Geometry["Shape"] == 'spherical':
         Geometry.pop('Height',None)

    # get sample volume in container
    space = GeometryFactory.factory(Geometry)
    Geometry.pop("Shape", None)
    volume_in_container = space.volume(**Geometry)

    print "NeXus Packing Fraction:",  info["mass"] / volume_in_container / info["mass_density"]
    # get packing fraction
    if PackingFraction is None:
        sample_density = info["mass"] / volume_in_container
        PackingFraction = sample_density / info["mass_density"]

    print "PackingFraction:", PackingFraction

    # get sample volume in the beam and correct mass density of what is in the beam
    if space.getShape() == 'Cylinder':
        Geometry["Height"] = BeamWidth
    volume_in_beam = space.volume(**Geometry)
    mass_density_in_beam = PackingFraction * info["mass_density"]


    # get molecular mass
    # Mantid SetSample doesn't set the actual height or radius. Have to use the setHeight, setRadius, ....
    ws = CreateSampleWorkspace()
    #SetSample(ws, Geometry={"Shape" : "Cylinder", "Height" : geo_dict["height"], "Radius" : geo_dict["radius"], "Center" : [0.,0.,0.]},
    #              Material={"ChemicalFormula" : info["formula"], "SampleMassDensity" : PackingFraction * info["mass_density"]})

    print info["formula"], mass_density_in_beam, volume_in_beam
    SetSampleMaterial(ws, ChemicalFormula=info["formula"], SampleMassDensity=mass_density_in_beam)
    material = ws.sample().getMaterial()

    # set constant
    avogadro =  6.022*10**23.

    # get total atoms and individual atom info
    natoms = sum([ x for x in material.chemicalFormula()[1] ])
    concentrations = { atom.symbol : {'concentration' : conc, 'mass' : atom.mass} for atom, conc in zip( material.chemicalFormula()[0], material.chemicalFormula()[1]) }
    neutron_info  = { atom.symbol : atom.neutron() for atom in material.chemicalFormula()[0] }
    atoms = combine_dictionaries(concentrations, neutron_info)

    sigfree = [ atom['tot_scatt_xs']*atom['concentration']*(atom['mass']/(atom['mass']+1.0))**2. for atom in atoms.values() ]
    print sum(sigfree)

    # get number of atoms using packing fraction, density, and volume
    print "Total scattering Xsection", material.totalScatterXSection() * natoms
    print "Coh Xsection:", material.cohScatterXSection() * natoms
    print "Incoh Xsection:", material.incohScatterXSection() * natoms
    print "Abs. Xsection:", material.absorbXSection() * natoms

    print ''.join( [x.strip() for x in info["formula"] ]), "#sample title"
    print info["formula"], "#sample formula"
    print info["mass_density"], "#density"
    print Geometry["Radius"], "#radius"
    print PackingFraction, "#PackingFraction"
    print space.getShape(), "#sample shape"
    print "nogo", "#do absorption correction now"
    print info["mass_density"]/ material.relativeMolecularMass() * avogadro / 10**24., "Sample density in form unit / A^3"

    print "\n\n#########################################################"
    print "##############Check levels###########################################"
    print "b bar:", material.cohScatterLengthReal()
    print "sigma:", material.totalScatterXSection()
    print "b: ", np.sqrt(material.totalScatterXSection()/(4.*np.pi))
    print material.cohScatterLengthReal() * material.cohScatterLengthReal() * natoms * natoms, "# (sum b)^2"
    print material.cohScatterLengthSqrd() * natoms, "# (sum c*bbar^2)"
    self_scat =  material.totalScatterLengthSqrd() * natoms / 100. # 100 fm^2 == 1 barn
    print "self scattering:", self_scat
    print "#########################################################\n"


    natoms_in_beam = mass_density_in_beam / material.relativeMolecularMass() * avogadro / 10**24. * volume_in_beam
    #print "Sample density (corrected) in form unit / A^3: ", mass_density_in_beam/ ws.sample().getMaterial().relativeMolecularMass() * avogadro / 10**24.
    return natoms_in_beam, self_scat


def calc_placzek_first_moment(q, mass):
    "Input: Q in Anstrom^-1 and Mass in AMU"
    amu_per_kg = 1. / physical_constants['atomic mass constant'][0] 
    ang_per_m = 1. / physical_constants['Angstrom star'][0]
    hbar_amu_ang2_per_s = hbar * amu_per_kg * ang_per_m * ang_per_m  # J*s -> amu*ang^2/s conversion
    hbar2 = hbar_amu_ang2_per_s * hbar_amu_ang2_per_s
    return ( hbar2 / 2.0 ) * ( q*q / mass / Avogadro)

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
    spline_fit = interpolate.UnivariateSpline(x_fit, y_fit, w=1./np.sqrt(var))
    spline_fit_prime = spline_fit.derivative()
    return spline_fit, spline_fit_prime 

def plotPlaczek(x, y, fit, fit_prime, title=None):
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
   

def calc_self_placzek( incident_ws, mass_amu, self_scat, theta, 
                       incident_path_length, scattered_path_length, detector='1/v'):
    # constants and conversions
    neutron_mass = m_n / physical_constants['atomic mass unit-kilogram relationship'][0]

    # variables
    l_0 = incident_path_length
    l_s = scattered_path_length
    l_total = l_0 + l_s

    angle_conv = np.pi / 180.
    sin_theta = np.sin(theta * angle_conv)

    # get derivative of incident spectrum using cubic spline
    incident_monitor = 0
    x = incident_ws.readX(incident_monitor)
    y = incident_ws.readY(incident_monitor)

    # Fit with Cubic Spline
    lam_lo = 0.15
    lam_hi = 3.2
    lam_lo = min(x)
    lam_hi = max(x)
    print "Wavelength:", lam_lo, 'to', lam_hi
    fit, fit_prime = fitCubicSpline(x, y, x_lo=lam_lo, x_hi=lam_hi)
    plotPlaczek(x, y, fit, fit_prime, title='Simple Cubic Spline: Default')

    fit, fit_prime = fitCubicSpline(x, y, x_lo=lam_lo, x_hi=lam_hi,s=1e5)
    plotPlaczek(x, y, fit, fit_prime, title='Simple Cubic Spline: Overfit')

    fit, fit_prime = fitCubicSpline(x, y, x_lo=lam_lo, x_hi=lam_hi,s=1e14)
    plotPlaczek(x, y, fit, fit_prime, title='Simple Cubic Spline: Underfit')

    # Fit with Howells Function
    fit, fit_prime = fitHowellsFunction(x, y, x_lo=lam_lo, x_hi=lam_hi)
    plotPlaczek(x, y, fit, fit_prime, title='HowellsFunction')

    spline_fit, spline_fit_prime =  fitCubicSplineWithGaussConv(x, y,x_lo=lam_lo,x_hi=lam_hi)
    fit = spline_fit(x)
    fit_prime = spline_fit_prime(x)
    plotPlaczek(x, y, fit, fit_prime, title='Cubic Spline w/ Gaussian Kernel Convolution ')

    if detector == '1/v':
        # See Powles (1973) Eq. (4.23)' for C
        d_ln_f_over_d_ln_lambda = spline_fit_prime(x)
        term_1 = (2.*l_0 + 3.*l_s) / l_total
        term_2 = (l_s / l_total) * d_ln_f_over_d_ln_lambda
        term_3 = (l_0 / l_total) * -1.
        C = term_1 + term_2 + term_3
        # See Powles (1973) Eq. (4.23)
        moment_1 = 2. * ( mass_amu / neutron_mass ) * sin_theta * sin_theta * C


    return self_scat * (1. - moment_1)


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



#-----------------------------------------------------------------------------------------#
#Get incident spectrum from vanadium

def plot_workspace(van, title=None,mode='TOF'):
    xlims = { 'TOF' : [0.,20000.],
              'Wavelength' : [0.,4.],
              'MomentumTransfer' : [0.,40.],
              'Energy' : [1.,1.e4]
            }

    mode = 'Wavelength'
    ws = ConvertUnits(InputWorkspace=van, Target=mode, EMode="Elastic")
    for bank in range(ws.getNumberHistograms()):
        xall = ws.readX(bank)[1:]
        yall = ws.readY(bank)

        if mode == 'Energy':
            plt.semilogx(xall, yall)
        else:
            plt.plot(xall, yall)

    plt.title(title)
    
    xunit = ws.getAxis(0).getUnit()
    plt.xlabel(str(xunit.caption())+'('+str(xunit.symbol())+')')

    yunit = ws.getAxis(1).getUnit()
    plt.ylabel('Neutrons per second per angstrom')

    axes = plt.gca()
    axes.set_xlim(xlims[mode])

    plt.show()
    return

def getIncidentSpectrumsFromVanadium( van, van_bg, binning, **alignAndFocusArgs):
    diameter_Vrod_cm = 0.585
    radius_Vrod_cm = diameter_Vrod_cm / 2.0
    mass_density_Vrod = 6.11
    height_Vrod_cm = 1.8
    print van
    nvan_atoms, tmp = getAbsScaleInfoFromNexus(van,
                                               PackingFraction=1.0,
                                               SampleMassDensity=mass_density_Vrod,
                                               Geometry={"Radius" : radius_Vrod_cm, "Height" : height_Vrod_cm},
                                               ChemicalFormula="V")

    van = ','.join(['NOM_%d' % num for num in van_scans])
    AlignAndFocusPowderFromFiles(OutputWorkspace='vanadium', Filename=van, AbsorptionWorkspace=None, **alignAndFocusArgs)
    van = 'vanadium'
    NormaliseByCurrent(InputWorkspace=van, OutputWorkspace=van,
                       RecalculatePCharge=True)
    SetSample(InputWorkspace=van,
                      Geometry={'Shape' : 'Cylinder', 'Height' : height_Vrod_cm,
                                'Radius' : radius_Vrod_cm, 'Center' : [0.,0.,0.]},
                      Material={'ChemicalFormula': 'V', 'SampleMassDensity' : mass_density_Vrod} )

    AlignAndFocusPowderFromFiles(OutputWorkspace='vanadium_background', Filename=van_bg, AbsorptionWorkspace=None, **alignAndFocusArgs)
    van_bg = 'vanadium_background'
    NormaliseByCurrent(InputWorkspace=van_bg, OutputWorkspace=van_bg,
                       RecalculatePCharge=True)

    plot_workspace(van, title='V')
    ConvertUnits(InputWorkspace=van, OutputWorkspace=van, Target="TOF", EMode="Elastic")

    Minus(LHSWorkspace=van, RHSWorkspace=van_bg, OutputWorkspace=van)
    plot_workspace(van, title='V - B')
    ConvertUnits(InputWorkspace=van, OutputWorkspace=van, Target="MomentumTransfer", EMode="Elastic")



    van_corrected = 'van_corrected'
    ConvertUnits(InputWorkspace=van, OutputWorkspace=van, Target="Wavelength", EMode="Elastic")
    if van_corr_type == 'Carpenter':
        MultipleScatteringCylinderAbsorption(InputWorkspace=van, OutputWorkspace=van_corrected, CylinderSampleRadius=radius_Vrod_cm)
    elif van_corr_type == 'Mayers':
        MayersSampleCorrection(InputWorkspace=van, OutputWorkspace=van_corrected, MultipleScattering=True) 
    else:
        print "NO VANADIUM absorption or multiple scattering!"

    plot_workspace(van_corrected, title='V-B (ms_abs corr.)')

    ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', EMode='Elastic')

    mtd[van_corrected] = (1./nvan_atoms)*mtd[van_corrected]
    ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', EMode='Elastic')

    plot_workspace(van_corrected, title='V-B (ms_abs corr.  + 1/N)')

    ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                 Target='dSpacing', EMode='Elastic')
    StripVanadiumPeaks(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                       BackgroundType='Quadratic')
    plot_workspace(van_corrected, title='V-B (ms_abs corr.  + 1/N + stripped)')
    ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                 Target='TOF', EMode='Elastic')
    FFTSmooth(InputWorkspace=van_corrected,
              OutputWorkspace=van_corrected,
              Filter="Butterworth",
              Params='20,2',
              IgnoreXBins=True,
              AllSpectra=True)
    plot_workspace(van_corrected, title='V-B (ms_abs corr.  + 1/N + stripped + smoothed)')
    ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', EMode='Elastic')
    Rebin(InputWorkspace=van_corrected, OutputWorkspace=van_corrected, Params=binning, PreserveEvents=False)

    return van_corrected

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

#-----------------------------------------------------------------------------------------#
# Start Placzek calculations

# get incident spectrums, conver to wavelength for proper fitting  and select bank (banks=0,1,2,3,4,5)
'''
incident_ws = getIncidentSpectrumsFromVanadium(van_scans, van_bg, binning, **alignAndFocusArgs)
ConvertUnits(InputWorkspace=incident_ws, OutputWorkspace=incident_ws,
             Target='TOF', EMode='Elastic')
'''
for binsize in [0.01,0.02,0.05,0.075]:
    lam_lo = 0.1
    lam_hi = 3.1
    lam_binning = ','.join([ str(x) for x in [lam_lo,binsize,lam_hi]])
    print "Wavelength Binning:", lam_binning
    incident_ws = getIncidentSpectrumFromMonitor(van_scans,lam_binning=lam_binning)
    SetSampleMaterial(incident_ws, ChemicalFormula='Si')
    self_scat = incident_ws.sample().getMaterial().totalScatterLengthSqrd() / 100.
    mass = incident_ws.sample().getMaterial().relativeMolecularMass()

    # Fitting using a cubic spline 
    theta = 120.4
    l_0 = 19.5
    l_1 = 1.11
    placzek_out =  calc_self_placzek( incident_ws, mass, self_scat, theta, l_0, l_1 )

    # Output the Placzek correction 
    import matplotlib.pyplot as plt

    #plt.plot(incident_ws.readX(0), placzek_out, 'x-')
    #plt.show()

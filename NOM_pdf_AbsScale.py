#!/usr/bin/env python
import os
import sys
import glob
import re
import json
import collections
import ConfigParser
from h5py import File
from mantid import mtd
from mantid.simpleapi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_n, micro
from scipy.constants import physical_constants
from scipy import interpolate, signal, ndimage, optimize

#-----------------------------------------------------------------------------------------#
# JSON load with convert from unicode to string

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data



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




def save_file(ws, title, header=list()):
    with open(title,'w') as f:
        for line in header:
            f.write('# %s \n' % line)
    SaveAscii(InputWorkspace=ws,Filename=title,Separator='Space',ColumnHeader=False,AppendToFile=True)

def save_banks(ws,title,binning=None ):
    CloneWorkspace(InputWorkspace=ws, OutputWorkspace="tmp")
    if binning is not None:
        Rebin(InputWorkspace="tmp", 
              OutputWorkspace="tmp", 
              Params=binning, 
              PreserveEvents=True)
    SaveAscii(InputWorkspace="tmp",
              Filename=title,
              Separator='Space',
              ColumnHeader=False,
              AppendToFile=False,
              SpectrumList=range(mtd["tmp"].getNumberHistograms()) )
    return

def save_banks_with_fit( title, fitrange_individual, InputWorkspace=None, **kwargs ):
    # Header
    for i, fitrange in enumerate(fitrange_individual):
        print 'fitrange:', fitrange[0], fitrange[1]

        Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
            WorkspaceIndex=i,
            StartX=fitrange[0], EndX=fitrange[1], # range cannot include area with NAN
            InputWorkspace=InputWorkspace, Output=InputWorkspace, OutputCompositeMembers=True)
        fitParams = mtd[InputWorkspace+'_Parameters']

        bank_title=title+'_'+InputWorkspace+'_bank_'+str(i)+'.dat'
        with open(bank_title,'w') as f:
            if 'btot_sqrd_avg' in kwargs:
                f.write('#<b^2> : %f \n' % kwargs['btot_sqrd_avg'])
            if 'bcoh_avg_sqrd' in kwargs:
                f.write('#<b>^2 : %f \n' % kwargs['bcoh_avg_sqrd'])
            if 'self_scat' in kwargs:
                f.write('#self scattering : %f \n' % kwargs['self_scat'])
            f.write('#fitrange: %f %f \n' % (fitrange[0], fitrange[1]))
            f.write('#for bank%d: %f + %f * Q\n' % (i+1, fitParams.cell('Value', 0), fitParams.cell('Value', 1)))

    # Body
    for bank in range(mtd[InputWorkspace].getNumberHistograms()):
        x_data = mtd[InputWorkspace].readX(bank)[0:-1]
        y_data = mtd[InputWorkspace].readY(bank)
        bank_title=title+'_'+InputWorkspace+'_bank_'+str(bank)+'.dat'
        print "####", bank_title
        with open(bank_title,'a') as f:
            for x, y in zip(x_data, y_data):
                f.write("%f %f \n" % (x, y))

def generateCropingTable(qmin, qmax):
    mask_info = CreateEmptyTableWorkspace()
    mask_info.addColumn("str", "SpectraList")
    mask_info.addColumn("double", "XMin")
    mask_info.addColumn("double", "XMax")
    for (i, value) in enumerate(qmin):
        mask_info.addRow([str(i), 0.0, value])
    for (i, value) in enumerate(qmax):
        mask_info.addRow([str(i), value, 100.0])

    return mask_info

def getQmaxFromData(Workspace=None, WorkspaceIndex=0):
    if Workspace is None:
        return None
    return max(mtd[Workspace].readX(WorkspaceIndex))

#-----------------------------------------------------------------------------------------
# Event Filters

def GenerateEventsFilterFromFiles(filenames, OutputWorkspace,
                                  InformationWorkspace, **kwargs):
    pass

def GenerateEventsFilterFromFiles(filenames, OutputWorkspace, InformationWorkspace, **kwargs):
    
    logName = kwargs.get('LogName', None)
    minValue = kwargs.get('MinimumLogValue', None)
    maxValue = kwargs.get('MaximumLogValue', None)
    logInterval = kwargs.get('LogValueInterval', None)
    unitOftime = kwargs.get('UnitOfTime', 'Nanoseconds')
    
    # TODO - handle multi-file filtering. Delete this line once implemented.
    assert len(filenames) == 1, 'ERROR: Multi-file filtering is not yet supported. (Stay tuned...)'
    
    for i, filename in enumerate(filenames):
        Load(Filename=filename, OutputWorkspace=filename)
        splitws, infows = GenerateEventsFilter(InputWorkspace=filename, 
                                               UnitOfTime=unitOfTime,
                                               LogName=logName, 
                                               MinimumLogValue=minValue, 
                                               MaximumLogValue=maxValue,
                                               LogValueInterval=logInterval )
        if i == 0:
            GroupWorkspaces( splitws, OutputWorkspace=Outputworkspace )
            GroupWorkspaces( infows, OutputWorkspace=InformationWorkspace )
        else:
            mtd[OutputWorkspace].add(splitws)
            mtd[InformationWorkspace].add(infows)
    return 

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

    info['packing_fraction'] = PackingFraction

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
    return natoms_in_beam, self_scat, info


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

def fitCubicSpline(x_fit, y_fit, x, s=1e15):
    tck = interpolate.splrep(x_fit,y_fit,s=s)
    fit = interpolate.splev(x,tck,der=0)
    fit_prime = interpolate.splev(x,tck,der=1)
    return fit, fit_prime

def fitCubicSplineViaMantidSplineSmoothing(InputWorkspace, Params, **kwargs):
    Rebin(InputWorkspace=InputWorkspace, OutputWorkspace='fit', Params=Params, PreserveEvents=True)
    SplineSmoothing(InputWorkspace='fit', OutputWorkspace='fit', OutputWorkspaceDeriv='fit_prime', DerivOrder=1,**kwargs)
    return mtd['fit'].readY(0), mtd['fit_prime_1'].readY(0)

def fitHowellsFunction(x_fit, y_fit, x ):
    # Fit with analytical function from HowellsEtAl
    def calc_HowellsFunction(lambdas, phi_max, phi_epi, lam_t, lam_1, lam_2, a ):
        term1 = phi_max * ((lam_t**4.)/lambdas**5.)*np.exp(-(lam_t/lambdas)**2.)
        term2 = (phi_epi/(lambdas**(1.+2.*a)))*(1./(1+np.exp((lambdas-lam_1)/lam_2)))
        return term1 + term2

    def calc_HowellsFunction1stDerivative(lambdas, phi_max, phi_epi, lam_t, lam_1, lam_2, a ):
        term1 = (((2*lam_t**2)/lambdas**2) - 5.) * (1./lambdas) * phi_max * ((lam_t**4.)/lambdas**5.)*np.exp(-(lam_t/lambdas)**2.) 
        term2 = ((1+2*a)/lambdas)*(1./lambdas)*(phi_epi/(lambdas**(1.+2.*a)))*(1./(1+np.exp((lambdas-lam_1)/lam_2)))
        return term1 + term2

    params = [1.,1.,1.,0.,1.,1.]
    params, convergence = optimize.curve_fit( calc_HowellsFunction, x_fit, y_fit, params)
    fit = calc_HowellsFunction(x, *params)
    fit_prime = calc_HowellsFunction1stDerivative(x, *params)
    return fit, fit_prime

def fitCubicSplineWithGaussConv(x_fit, y_fit, x, sigma=3):
    # Fit with Cubic Spline using a Gaussian Convolution to get weights
    def moving_average(y, sigma=sigma):
        b = signal.gaussian(39, sigma)
        average = ndimage.filters.convolve1d(y, b/b.sum())
        var = ndimage.filters.convolve1d(np.power(y-average,2),b/b.sum())
        return average, var

    avg, var = moving_average(y_fit)
    spline_fit = interpolate.UnivariateSpline(x_fit, y_fit, w=1./np.sqrt(var))
    spline_fit_prime = spline_fit.derivative()
    fit = spline_fit(x)
    fit_prime = spline_fit_prime(x)
    return fit, fit_prime


#-----------------------------------------------------------------------------------------#
#Get incident spectrum from Monitor 

def plotIncidentSpectrum(x, y, x_fit, fit, fit_prime, title=None):
    plt.plot(x,y,'bo',x_fit,fit,'--')
    plt.legend(['Incident Spectrum','Fit f(x)'],loc='best')
    if title is not None:
        plt.title(title)
    plt.show()

    plt.plot(x_fit,fit_prime/fit,'x--',label="Fit f'(x)/f(x)")
    plt.xlabel('Wavelength')
    plt.legend()
    if title is not None:
        plt.title(title)
    axes = plt.gca()
    axes.set_ylim([-12,6])
    plt.show()
    return


def GetIncidentSpectrumFromMonitor(Filename, OutputWorkspace="IncidentWorkspace",
                                   IncidentIndex=0, TransmissionIndex=1, 
                                   LambdaBinning="-6000",
                                   BinType="ResampleX"):

    Filename = str(Filename)

    #-------------------------------------------------
    # Joerg's read_bm.pro code

    # Loop workspaces to get each incident spectrum
    monitor_raw = LoadNexusMonitors(Filename)
    monitor = 'monitor'
    NormaliseByCurrent(InputWorkspace=monitor_raw, OutputWorkspace=monitor,
                       RecalculatePCharge=True)
    ConvertUnits(InputWorkspace=monitor, OutputWorkspace=monitor,
                 Target='Wavelength', EMode='Elastic')
    if BinType == 'ResampleX':
        LambdaBinning = int(LambdaBinning)
        monitor = ResampleX(InputWorkspace=monitor, 
                            XMin=0.1,
                            XMax=2.9,
                            NumberBins=abs(LambdaBinning), 
                            LogBinning=(LambdaBinning < 0), 
                            PreserveEvents=True)
    elif BinType == 'Rebin':
        monitor = Rebin(InputWorkspace=monitor, 
                        Params=LambdaBinning, 
                        PreserveEvents=True)

    lam = monitor.readX(IncidentIndex)[:-1] # wavelength in A
    bm  = monitor.readY(IncidentIndex)     # neutron counts / microsecond
    p = 0.0000794807
    abs_xs_3He = 5333.0                   # barns for lambda == 1.8 A
    e0 = abs_xs_3He * lam / 1.8 * 2.43e-5 * p # p is set to give efficiency of 1.03 10^-5 at 1.8 A
    bmeff = bm / ( 1. - np.exp(-e0))      # neutron counts / microsecond
    bmeff = bmeff / micro                 # neutron counts / second
    
    CreateWorkspace(DataX=lam, DataY=bmeff, 
                    OutputWorkspace=OutputWorkspace, UnitX='Wavelength')
    mtd[OutputWorkspace].setYUnit('Counts')
    return mtd[OutputWorkspace]

def FitIncidentSpectrum(InputWorkspace, OutputWorkspace,FitSpectrumWith='GaussConvCubicSpline', BinningForFit="0.15,0.05,3.2"):

    incident_ws = mtd[InputWorkspace]

    # Fit Incident Spectrum  
    incident_index = 0
    x = incident_ws.readX(incident_index)
    y = incident_ws.readY(incident_index)

    Rebin(incident_ws, OutputWorkspace='fit', Params=BinningForFit, PreserveEvents=True)
    x_fit = np.array(mtd['fit'].readX(incident_index))
    y_fit = np.array(mtd['fit'].readY(incident_index))

    if FitSpectrumWith == 'CubicSpline':
        fit, fit_prime = fitCubicSpline(x_fit, y_fit, x, s=1e7)
        plotIncidentSpectrum(x_fit, y_fit, x, fit, fit_prime, title='Simple Cubic Spline: Default')

    elif FitSpectrumWith == 'CubicSplineViaMantid':
        fit, fit_prime = fitCubicSplineViaMantidSplineSmoothing(InputWorkspace, Params=BinningForFit, MaxNumberOfBreaks=8)
        plotIncidentSpectrum(x_fit, y_fit, x, fit, fit_prime, title='Cubic Spline via Mantid SplineSmoothing')

    elif FitSpectrumWith == 'HowellsFunction':
        fit, fit_prime = fitHowellsFunction(x_fit, y_fit, x)
        plotIncidentSpectrum(x_fit, y_fit, x, fit, fit_prime, title='HowellsFunction')

    elif FitSpectrumWith == 'GaussConvCubicSpline':
        fit, fit_prime =  fitCubicSplineWithGaussConv(x_fit, y_fit, x, sigma=2)
        plotIncidentSpectrum(x_fit, y_fit, x, fit, fit_prime, title='Cubic Spline w/ Gaussian Kernel Convolution ')

    else:
        raise Exception("Unknown method for fitting incident spectrum")
        return

    CreateWorkspace(DataX=x, DataY=np.append(fit,fit_prime), OutputWorkspace=OutputWorkspace, UnitX='Wavelength', NSpec=2)
    return mtd[OutputWorkspace]


#-----------------------------------------------------------------------------------------
# Placzek - 1st order inelastic correction

def plotPlaczek(x, y, fit, fit_prime, title=None):
    plt.plot(x,y,'bo',x,fit,'--')
    plt.legend(['Incident Spectrum','Fit f(x)'],loc='best')
    if title is not None:
        plt.title(title)
    plt.show()

    plt.plot(x,x*fit_prime/fit,'x--',label="Fit x*f'(x)/f(x)")
    plt.xlabel('Wavelength')
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()
    return


def GetLogBinning(start, stop, num=100):
    return  np.logspace(np.log(start), np.log(stop), num=num, endpoint=True, base=np.exp(1))

def ConvertLambdaToQ(lam,angle):
    angle_conv = np.pi / 180.
    sin_theta_by_2 = np.sin(angle * angle_conv / 2.)
    q = (4.*np.pi / lam)*sin_theta_by_2
    return q

def ConvertQToLambda(q,angle):
    angle_conv = np.pi / 180.
    sin_theta_by_2 = np.sin(angle * angle_conv / 2.)
    lam = (4.*np.pi / q)*sin_theta_by_2
    return lam


def CalculatePlaczekSelfScattering(IncidentWorkspace, ParentWorkspace, OutputWorkspace, 
                                   L1, L2, Polar, Azimuthal=None, Detector=None):

    # constants and conversions
    neutron_mass = m_n / physical_constants['atomic mass unit-kilogram relationship'][0]

    # get sample information: mass, total scattering length, and concentration of each species
    total_stoich = 0.0
    material =  mtd[IncidentWorkspace].sample().getMaterial().chemicalFormula()
    atom_species = collections.OrderedDict()
    for atom, stoich in zip(material[0], material[1]):
        totalScattLength = atom.neutron()['tot_scatt_length'] / 10.
        atom_species[atom.symbol] = {'mass' : atom.mass,
                                    'stoich' : stoich,
                                    'tot_scatt_length' : totalScattLength }
        total_stoich += stoich

    for atom, props in atom_species.iteritems():
        props['concentration'] = props['stoich'] / total_stoich

    # calculate summation term w/ neutron mass over molecular mass ratio
    summation_term = 0.0
    for species, props in atom_species.iteritems():
        summation_term += props['concentration'] * props['tot_scatt_length'] * neutron_mass / props['mass']

    # calculate elastic self-scattering term
    elastic_term = 0.0
    for species, props in atom_species.iteritems():
        elastic_term += props['concentration'] * props['tot_scatt_length']

    # get incident spectrum and 1st derivative 
    incident_index = 0
    incident_prime_index = 1

    x_lambda = mtd[IncidentWorkspace].readX(incident_index)
    incident = mtd[IncidentWorkspace].readY(incident_index)
    incident_prime = mtd[IncidentWorkspace].readY(incident_prime_index)

    phi_1 = x_lambda * incident_prime / incident

    # Set default Detector Law
    if Detector is None:
        Detector={'Alpha' : None,
                  'LambdaD' : 1.44,
                  'Law' : '1/v'}

    # Set detector exponential coefficient alpha
    if Detector['Alpha'] is None:
        Detector['Alpha'] = 2.* np.pi / Detector['LambdaD']

    # Detector law to get eps_1(lambda) 
    if Detector['Law'] == '1/v':
        c = -Detector['Alpha'] / (2. * np.pi)
        x = x_lambda
        detector_law_term = c*x*np.exp(c*x) / (1. - np.exp(c*x))

    eps_1 = detector_law_term

    # Set default azimuthal angle
    if Azimuthal is None:
        Azimuthal = np.zeros(len(Polar))
    # Placzek
    x_lambdas = np.array([])
    placzek_correction = np.array([])
    for bank, (l2, theta, phi) in enumerate(zip(L2, Polar, Azimuthal)):
        # variables
        L_total = L1 + l2
        f = L1 / L_total

        angle_conv = np.pi / 180.
        sin_theta = np.sin(theta * angle_conv)
        sin_theta_by_2 = np.sin(theta * angle_conv / 2.)

        term1 = (f - 1.) * phi_1
        term2 = f*eps_1
        term3 = f - 3.

        #per_bank_q = ConvertLambdaToQ(x_lambda,theta)
        per_bank_correction = 2.*(term1 - term2 + term3) * sin_theta_by_2 * sin_theta_by_2 * summation_term
        #per_bank_correction = elastic_term + 2.*(term1 - term2 + term3) * sin_theta_by_2 * sin_theta_by_2 * summation_term
        x_lambdas = np.append(x_lambdas, x_lambda)
        placzek_correction = np.append(placzek_correction, per_bank_correction)

    CreateWorkspace(DataX=x_lambdas, DataY=placzek_correction, OutputWorkspace=OutputWorkspace,
                    UnitX='Wavelength',  NSpec=len(Polar), ParentWorkspace=ParentWorkspace)
    
    return mtd[OutputWorkspace]


def print_unit_info(workspace):
    ws = mtd[workspace]
    for i in range(ws.axes()):
        axis = ws.getAxis(i)
        print "Axis {0} is a {1}{2}{3}".format(i,
                                           "Spectrum Axis" if axis.isSpectra() else "",
                                           "Text Axis" if axis.isText() else "",
                                           "Numeric Axis" if axis.isNumeric() else "")

        unit = axis.getUnit()
        print "\n YUnit:{0}".format(ws.YUnit())
        print "\t caption:{0}".format(unit.caption())
        print "\t symbol:{0}".format(unit.symbol())
    return


def SetInelasticCorrection(inelastic_dict):
    if inelastic_dict is None:
        inelastic_dict = {"Type" : None }
        return inelastic_dict

    corr_type = inelastic_dict["Type"]

    if corr_type == "Placzek":
        default_settings = {"Order" : "1st",
                            "Self" : "True",
                             "Interference" : "False",
                             "FitSpectrumWith" : "GaussConvCubicSpline",
                             "LambdaBinning" : "0.16,0.04,2.8"}
        inelastic_settings = default_settings.copy()
        inelastic_settings.update(inelastic_dict)

    else:
        raise Exception("Unknown Inelastic Correction Type")

    return inelastic_settings



#-----------------------------------------------------------------------------------
# MAIN - NOM_pdf


if "__main__" == __name__:
    configfile = sys.argv[1]
    print "loading config from", configfile
    with open(configfile) as handle:
        config = json_loads_byteified(handle.read())

    print config, type(config)
    title = config['title']

    # Get sample info
    sample = config['sam']
    sam_mass_density = sample.get('MassDensity', None)
    sam_packing_fraction = sample.get('PackingFraction', None)
    sam_geometry = sample.get('Geometry', None)
    sam_material = sample.get('Material', None)

    # Get normalization info
    van = config['van']
    van_mass_density = van.get('MassDensity', None)
    van_packing_fraction = van.get('PackingFraction',1.0)
    van_geometry = van.get('Geometry', None)
    van_material = van.get('Material', 'V')
 
    # Get calibration, characterization, and other settings
    calib = config['calib']
    charac = config['charac']
    binning= config['binning']
    high_q_linear_fit_range = config['high_q_linear_fit_range']
    wkspIndices=config['sumbanks'] # workspace indices - zero indexed arrays
    cache_dir = config.get("CacheDir", os.path.abspath('.'))

    # Create Nexus file basenames
    sam_scans = ','.join(['NOM_%d' % num for num in sample['Runs']])
    can = ','.join(['NOM_%d' % num for num in sample['Background']])
    van_scans = ','.join(['NOM_%d' % num for num in van['Runs']])
    van_bg = ','.join(['NOM_%d' % num for num in van['Background']])


    # Get absolute scale information from Nexus file
    print "#-----------------------------------#"
    print "# Sample"
    print "#-----------------------------------#"
    natoms, self_scat, sam_info = getAbsScaleInfoFromNexus(sample['Runs'],
                                                 PackingFraction=sam_packing_fraction,
                                                 SampleMassDensity=sam_mass_density,
                                                 Geometry=sam_geometry, 
                                                 ChemicalFormula=sam_material)

    print "#-----------------------------------#"
    print "# Vanadium"
    print "#-----------------------------------#"
    nvan_atoms, tmp, van_info = getAbsScaleInfoFromNexus(van['Runs'],
                                               PackingFraction=1.0,
                                               SampleMassDensity=van_mass_density,
                                               Geometry=van_geometry,
                                               ChemicalFormula="V")

    print "Sample natoms:", natoms
    print "Vanadium natoms:", nvan_atoms
    print "Vanadium natoms / Sample natoms:", nvan_atoms/natoms
    print

    # Get sample corrections
    sam_mass_density = float(sample.get('MassDensity', sam_info['mass_density']))
    sam_material = sample.get('Material', sam_info['formula'])
    sam_packing_fraction = sample.get('PackingFraction',sam_info['packing_fraction'])
    sam_geometry = sample.get('Geometry', None)
    sam_abs_corr= sample.get("AbsorptionCorrection", "Carpenter")
    sam_ms_corr = sample.get("MultipleScatteringCorrection", "Carpenter")
    sam_inelastic_corr = SetInelasticCorrection(sample.get('InelasticCorrection', None))

    # Get vanadium corrections
    van_material = van.get('Material', 'V')
    van_mass_density = van.get('MassDensity', van_info['mass_density'])
    van_packing_fraction = van.get('PackingFraction', van_info['packing_fraction'])
    van_geometry = van.get('Geometry', None)
    van_abs_corr = van.get("AbsorptionCorrection", "Carpenter")
    van_ms_corr = van.get("MultipleScatteringCorrection", "Carpenter")
    van_inelastic_corr = SetInelasticCorrection(van.get('InelasticCorrection', None))
 
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

    # TODO take out the RecalculatePCharge in the future once tested

    #-----------------------------------------------------------------------------------------#
    # Load Sample
    AlignAndFocusPowderFromFiles(OutputWorkspace='sample', 
                                 Filename=sam_scans, 
                                 Absorption=None, 
                                 **alignAndFocusArgs)
    sam = 'sample'
    NormaliseByCurrent(InputWorkspace=sam, 
                       OutputWorkspace=sam,
                       RecalculatePCharge=True)
    #SaveNexusProcessed(mtd[sam], os.path.abspath('.') + '/sample_nexus.nxs')
    ConvertUnits(InputWorkspace=sam, 
                 OutputWorkspace=sam, 
                 Target="MomentumTransfer", 
                  EMode="Elastic")
    save_banks(sam, title="sample_and_can.dat", binning=binning)

    #-----------------------------------------------------------------------------------------#
    # Load Sample Container
    AlignAndFocusPowderFromFiles(OutputWorkspace='container', 
                                 Filename=can, 
                                 Absorption=None, 
                                 **alignAndFocusArgs)
    can = 'container'
    NormaliseByCurrent(InputWorkspace=can, 
                       OutputWorkspace=can,
                       RecalculatePCharge=True)
    #SaveNexusProcessed(mtd['container'], os.path.abspath('.') + '/container_nexus.nxs')
    ConvertUnits(InputWorkspace=can, 
                 OutputWorkspace=can, 
                 Target="MomentumTransfer", 
                 EMode="Elastic")
    save_banks(can, title="can.dat", binning=binning)


    #-----------------------------------------------------------------------------------------#
    # Load Vanadium 
    #Load(Filename=van_abs, OutputWorkspace='van_absorption')
    AlignAndFocusPowderFromFiles(OutputWorkspace='vanadium', 
                                 Filename=van_scans, 
                                 AbsorptionWorkspace=None, 
                                 **alignAndFocusArgs)
    van_wksp = 'vanadium'
    van_geometry.update( {'Shape' : 'Cylinder', 'Center' : [0.,0.,0.,] } )
    NormaliseByCurrent(InputWorkspace=van_wksp, 
                       OutputWorkspace=van_wksp,
                       RecalculatePCharge=True)
    SetSample(InputWorkspace=van_wksp, 
              Geometry=van_geometry,
              Material={'ChemicalFormula': van_material, 'SampleMassDensity' : van_mass_density} )
    #SaveNexusProcessed(mtd['vanadium'], os.path.abspath('.') + '/vanadium_nexus.nxs')
    ConvertUnits(InputWorkspace=van_wksp, 
                 OutputWorkspace=van_wksp, 
                 Target="MomentumTransfer", 
                 EMode="Elastic")
    save_banks(van_wksp, title="vanadium_and_background.dat", binning=binning)


    #-----------------------------------------------------------------------------------------#
    # Load Vanadium Background
    AlignAndFocusPowderFromFiles(OutputWorkspace='vanadium_background', 
                                 Filename=van_bg, 
                                 AbsorptionWorkspace=None, 
                                 **alignAndFocusArgs)

    van_bg = 'vanadium_background'
    NormaliseByCurrent(InputWorkspace=van_bg, 
                       OutputWorkspace=van_bg,
                       RecalculatePCharge=True)
    #SaveNexusProcessed(mtd['vanadium_background'], os.path.abspath('.') + '/vanadium_background_nexus.nxs')
    ConvertUnits(InputWorkspace=van_bg, 
                 OutputWorkspace=van_bg, 
                 Target="MomentumTransfer", 
                 EMode="Elastic")
    save_banks(van_bg, title="vanadium_background.dat", binning=binning)


    #-----------------------------------------------------------------------------------------#
    # Load Instrument Characterizations
    PDDetermineCharacterizations(InputWorkspace=sam,
                                 Characterizations='characterizations',
                                 ReductionProperties='__snspowderreduction')
    propMan = PropertyManagerDataService.retrieve('__snspowderreduction')
    qmax = 2.*np.pi/propMan['d_min'].value
    qmin = 2.*np.pi/propMan['d_max'].value
    for a,b in zip(qmin, qmax):
        print 'Qrange:', a, b
    mask_info = generateCropingTable(qmin, qmax)


    #-----------------------------------------------------------------------------------------#
    # STEP 1: Subtract Backgrounds 

    Minus(LHSWorkspace=sam, RHSWorkspace=can, OutputWorkspace=sam)
    Minus(LHSWorkspace=van_wksp, RHSWorkspace=van_bg, OutputWorkspace=van_wksp)

    ConvertUnits(InputWorkspace=van_wksp, 
                 OutputWorkspace=van_wksp, 
                 Target="MomentumTransfer", 
                 EMode="Elastic")
    save_banks(van_wksp, title="vanadium_minus_background.dat", binning=binning)

    #-----------------------------------------------------------------------------------------#
    # STEP 2.0: Prepare vanadium as normalization calibrant

    # Multiple-Scattering and Absorption (Steps 2-4) for Vanadium

    van_corrected = 'van_corrected'
    ConvertUnits(InputWorkspace=van_wksp, 
                 OutputWorkspace=van_corrected, 
                 Target="Wavelength", 
                 EMode="Elastic")

    if van_abs_corr['Type'] == 'Carpenter' or van_ms_corr['Type'] == 'Carpenter':
        MultipleScatteringCylinderAbsorption(InputWorkspace=van_corrected, 
                                             OutputWorkspace=van_corrected, 
                                             CylinderSampleRadius=van['Geometry']['Radius'])
    elif van_abs_corr['Type'] == 'Mayers' or van_ms_corr['Type'] == 'Mayers':
        if van_ms_corr['Type'] == 'Mayers':
            MayersSampleCorrection(InputWorkspace=van_corrected, 
                                   OutputWorkspace=van_corrected, 
                                   MultipleScattering=True) 
        else:
            MayersSampleCorrection(InputWorkspace=van_corrected, 
                                   OutputWorkspace=van_corrected, 
                                   MultipleScattering=False) 
    else:
        print "NO VANADIUM absorption or multiple scattering!"

    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', 
                 EMode='Elastic')
    van_title = "vanadium_ms_abs_corrected"
    save_banks(van_corrected, title=van_title+'.dat', binning=binning)

    # Divide by numer of vanadium atoms (Step 5)
    mtd[van_corrected] = (1./nvan_atoms)*mtd[van_corrected]
    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', 
                 EMode='Elastic')
    van_title += '_norm_by_atoms'
    save_banks(van_corrected, title=van_title+".dat", binning=binning)

    # Divide by total scattering length squared = total scattering cross-section over 4 * pi
    sigma_v = mtd[van_corrected].sample().getMaterial().totalScatterXSection()
    prefactor = ( sigma_v / (4.*np.pi) )
    mtd[van_corrected] = (1./prefactor)*mtd[van_corrected]
    van_title += '_multiply_by_tot_scat_len'
    save_banks(van_corrected, title=van_title+"_with_peaks.dat", binning=binning)

    # TODO subtract self-scattering of vanadium (According to Eq. 7 of Howe, McGreevey, and Howells, JPCM, 1989)

    # Smooth Vanadium (strip peaks plus smooth)

    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='dSpacing', 
                 EMode='Elastic')
    StripVanadiumPeaks(InputWorkspace=van_corrected, 
                       OutputWorkspace=van_corrected,
                       BackgroundType='Quadratic')
    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', 
                 EMode='Elastic')
    van_title += '_peaks_stripped'
    save_banks(van_corrected, title=van_title+".dat", binning=binning)

    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='TOF', 
                 EMode='Elastic')
    FFTSmooth(InputWorkspace=van_corrected,
              OutputWorkspace=van_corrected,
              Filter="Butterworth",
              Params='20,2',
              IgnoreXBins=True,
              AllSpectra=True)
    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', 
                 EMode='Elastic')
    van_title += '_smoothed'
    save_banks(van_corrected, title=van_title+".dat", binning=binning)

    # debugging
    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='Wavelength', 
                 EMode='Elastic')

    # Inelastic correction
    print van_inelastic_corr['Type']
    if van_inelastic_corr['Type'] == "Placzek":
        for van_scan in van['Runs']:
            van_incident_wksp = 'van_incident_wksp'
            lambda_binning = van['InelasticCorrection']['LambdaBinning']
            GetIncidentSpectrumFromMonitor(van_scan, OutputWorkspace=van_incident_wksp) 

            fit_type = sample['InelasticCorrection']['FitSpectrumWith']
            FitIncidentSpectrum(InputWorkspace=van_incident_wksp, 
                                OutputWorkspace=van_incident_wksp,
                                FitSpectrumWith=fit_type,
                                BinningForFit=lambda_binning)

            van_placzek = 'van_placzek'
            
            SetSample(InputWorkspace=van_incident_wksp, 
                      Material={'ChemicalFormula': van_material, 
                                'SampleMassDensity' : van_mass_density} )
            CalculatePlaczekSelfScattering(IncidentWorkspace=van_incident_wksp,
                                           ParentWorkspace=van_corrected,
                                           OutputWorkspace=van_placzek,
                                           L1=19.5,
                                           L2=alignAndFocusArgs['L2'],
                                           Polar=alignAndFocusArgs['Polar'])
            ConvertUnits(InputWorkspace=van_placzek, 
                         OutputWorkspace=van_placzek,
                         Target='MomentumTransfer', 
                         EMode='Elastic')

        save_banks(van_placzek, title="vanadium_placzek.dat")

        ConvertUnits(InputWorkspace=van_corrected, 
                     OutputWorkspace=van_corrected,
                     Target='MomentumTransfer', 
                     EMode='Elastic')
        #ConvertToHistogram(InputWorkspace=van_placzek, 
        #                   OutputWorkspace=van_placzek)
        RebinToWorkspace(WorkspaceToRebin=van_corrected, 
                         WorkspaceToMatch=van_placzek, 
                         OutputWorkspace=van_corrected, 
                         PreserveEvents=True)
        
        Minus(LHSWorkspace=van_corrected, RHSWorkspace=van_placzek, OutputWorkspace=van_corrected)
        ConvertUnits(InputWorkspace=van_corrected, 
                     OutputWorkspace=van_corrected,
                     Target='MomentumTransfer', 
                     EMode='Elastic')
        van_title += '_placzek_corrected'
        save_banks(van_corrected, title=van_title+".dat", binning=binning)

    ConvertUnits(InputWorkspace=van_corrected, 
                 OutputWorkspace=van_corrected,
                 Target='MomentumTransfer', 
                 EMode='Elastic')
 
    SetUncertainties(InputWorkspace=van_corrected, 
                     OutputWorkspace=van_corrected,
                     SetError='zero')

    #-----------------------------------------------------------------------------------------#
    # STEP 2.1: Normalize by Vanadium


    for name in [sam, can, van_wksp, van_corrected, van_bg]:
        ConvertUnits(InputWorkspace=name, OutputWorkspace=name,
                     Target='MomentumTransfer', EMode='Elastic')

    save_banks(sam, title="sample_minus_back.dat", binning=binning)
    Divide(LHSWorkspace=sam, RHSWorkspace=van_corrected, OutputWorkspace=sam)
    sam_title = "sample_minus_back_normalized"
    save_banks(sam, title=sam_title+".dat", binning=binning)

    #-----------------------------------------------------------------------------------------#
    # STEP 3 & 4: Subtract multiple scattering and apply absorption correction

    ConvertUnits(InputWorkspace=sam, OutputWorkspace=sam, Target="Wavelength", EMode="Elastic")

    sam_corrected = 'sam_corrected'
    if sam_abs_corr['Type'] == 'Carpenter' or sam_ms_corr['Type'] == 'Carpenter':
        MultipleScatteringCylinderAbsorption(InputWorkspace=sam, 
                                             OutputWorkspace=sam_corrected, 
                                             CylinderSampleRadius=sample['Geometry']['Radius'])
    elif sam_abs_corr['Type'] == 'Mayers' or sam_ms_corr['Type'] == 'Mayers':
        if sam_ms_corr['Type'] == 'Mayers':
            MayersSampleCorrection(InputWorkspace=sam, 
                                   OutputWorkspace=sam_corrected, 
                                   MultipleScattering=True) 
        else:
            MayersSampleCorrection(InputWorkspace=sam, 
                                   OutputWorkspace=sam_corrected, 
                                   MultipleScattering=False) 
    else:
        print "NO SAMPLE absorption or multiple scattering!"
        CloneWorkspace(InputWorkspace=sam, OutputWorkspace=sam_corrected)

    ConvertUnits(InputWorkspace=sam_corrected, OutputWorkspace=sam_corrected,
                 Target='MomentumTransfer', EMode='Elastic')
    sam_title += "_ms_abs_corrected"
    save_banks(sam_corrected, title=sam_title+".dat", binning=binning)

    #-----------------------------------------------------------------------------------------#
    # STEP 5: Divide by number of atoms in sample

    mtd[sam_corrected] = (1./natoms) * mtd[sam_corrected]
    ConvertUnits(InputWorkspace=sam_corrected, OutputWorkspace=sam_corrected,
                 Target='MomentumTransfer', EMode='Elastic')
    sam_title += "_norm_by_atoms"
    save_banks(sam_corrected, title=sam_title+".dat", binning=binning)

    #-----------------------------------------------------------------------------------------#
    # STEP 6: Inelastic correction
    if sam_inelastic_corr['Type'] == "Placzek":
        for sam_scan in sample['Runs']:
            sam_incident_wksp = 'sam_incident_wksp'
            lambda_binning = sample['InelasticCorrection']['LambdaBinning']
            GetIncidentSpectrumFromMonitor(sam_scan, OutputWorkspace=sam_incident_wksp)

            fit_type = sample['InelasticCorrection']['FitSpectrumWith']
            FitIncidentSpectrum(InputWorkspace=sam_incident_wksp, 
                                OutputWorkspace=sam_incident_wksp,
                                FitSpectrumWith=fit_type,
                                BinningForFit=lambda_binning)

            sam_placzek = 'sam_placzek'
            SetSample(InputWorkspace=sam_incident_wksp, 
                      Material={'ChemicalFormula': sam_material, 
                                'SampleMassDensity' : sam_mass_density} )
            CalculatePlaczekSelfScattering(IncidentWorkspace=sam_incident_wksp,
                                           ParentWorkspace=sam_corrected,
                                           OutputWorkspace=sam_placzek,
                                           L1=19.5,
                                           L2=alignAndFocusArgs['L2'],
                                           Polar=alignAndFocusArgs['Polar'])
            ConvertUnits(InputWorkspace=sam_placzek, 
                         OutputWorkspace=sam_placzek,
                         Target='MomentumTransfer', 
                         EMode='Elastic')

        save_banks(sam_placzek, title="sample_placzek.dat")

        ConvertUnits(InputWorkspace=sam_corrected, 
                     OutputWorkspace=sam_corrected,
                     Target='MomentumTransfer', 
                     EMode='Elastic')
        ConvertToHistogram(InputWorkspace=sam_placzek, 
                           OutputWorkspace=sam_placzek)
        RebinToWorkspace(WorkspaceToRebin=sam_corrected, 
                         WorkspaceToMatch=sam_placzek, 
                         OutputWorkspace=sam_corrected, 
                         PreserveEvents=False)
        
        ConvertFromDistribution(sam_corrected)

        save_banks(sam_placzek, title="sample_placzek_v2.dat")
        save_banks(sam_corrected, title="sample_before_placzek_v2.dat")
        Minus(LHSWorkspace=sam_corrected, RHSWorkspace=sam_placzek, OutputWorkspace=sam_corrected)
        ConvertUnits(InputWorkspace=sam_corrected, 
                     OutputWorkspace=sam_corrected,
                     Target='MomentumTransfer', 
                     EMode='Elastic')
        sam_title += '_placzek_corrected'
        save_banks(sam_corrected, title=sam_title+".dat", binning=binning)


    #-----------------------------------------------------------------------------------------#
    # STEP 7: Output spectrum

    # TODO Since we already went from Event -> 2D workspace, can't use this anymore
    #if alignAndFocusArgs['PreserveEvents']:
    #    CompressEvents(InputWorkspace=sam_corrected, OutputWorkspace=sam_corrected)
    #    CompressEvents(InputWorkspace=van_corrected, OutputWorkspace=van_corrected)


    # S(Q) bank-by-bank Section
    material = mtd[sam].sample().getMaterial()
    bcoh_avg_sqrd = material.cohScatterLength()*material.cohScatterLength()
    btot_sqrd_avg = material.totalScatterLengthSqrd()
    print bcoh_avg_sqrd, btot_sqrd_avg
    term_to_subtract = btot_sqrd_avg / bcoh_avg_sqrd
    CloneWorkspace(InputWorkspace=sam_corrected, OutputWorkspace='SQ_banks_ws')
    SQ_banks =  (1./bcoh_avg_sqrd)*mtd['SQ_banks_ws'] - (term_to_subtract-1.) 


    # F(Q) bank-by-bank Section
    CloneWorkspace(InputWorkspace=sam_corrected, OutputWorkspace='FQ_banks_ws')
    FQ_banks_raw = mtd['FQ_banks_ws']
    FQ_banks = FQ_banks_raw - self_scat 

    #-----------------------------------------------------------------------------------------#
    # Ouput bank-by-bank with linear fits for high-Q 

    # fit the last 80% of the bank being used
    for i, q in zip(range(mtd[sam_corrected].getNumberHistograms()), qmax):
        qmax_data = getQmaxFromData(sam_corrected, i)
        qmax[i] = q if q <= qmax_data else qmax_data

    fitrange_individual = [(high_q_linear_fit_range*q, q) for q in qmax]

    for q in qmax:
        print 'Linear Fit Qrange:', high_q_linear_fit_range*q, q


    kwargs = { 'btot_sqrd_avg' : btot_sqrd_avg,
               'bcoh_avg_sqrd' : bcoh_avg_sqrd,
               'self_scat' : self_scat }

    '''
    save_banks_with_fit( title, fitrange_individual, InputWorkspace='SQ_banks', **kwargs)
    save_banks_with_fit( title, fitrange_individual, InputWorkspace='FQ_banks', **kwargs)
    save_banks_with_fit( title, fitrange_individual, InputWorkspace='FQ_banks_raw', **kwargs)
    '''
    save_banks('SQ_banks',         title=title+"_SQ_banks.dat",     binning=binning)
    save_banks('FQ_banks',         title=title+"_FQ_banks.dat",     binning=binning)
    save_banks('FQ_banks_raw', title=title+"_FQ_banks_raw.dat", binning=binning)

    #-----------------------------------------------------------------------------------------#
    # Event workspace -> Histograms
    Rebin(InputWorkspace=sam_corrected, OutputWorkspace=sam_corrected, Params=binning, PreserveEvents=True)
    Rebin(InputWorkspace=van_corrected, OutputWorkspace=van_corrected, Params=binning, PreserveEvents=True)
    Rebin(InputWorkspace='container',   OutputWorkspace='container',   Params=binning, PreserveEvents=True)
    Rebin(InputWorkspace='sample',      OutputWorkspace='sample',      Params=binning, PreserveEvents=True)
    Rebin(InputWorkspace=van_bg,        OutputWorkspace='background',      Params=binning, PreserveEvents=True)

    #-----------------------------------------------------------------------------------------#
    # Apply Qmin Qmax limits

    #MaskBinsFromTable(InputWorkspace=sam_corrected, OutputWorkspace='sam_single',       MaskingInformation=mask_info)
    #MaskBinsFromTable(InputWorkspace=van_corrected, OutputWorkspace='van_single',       MaskingInformation=mask_info)
    #MaskBinsFromTable(InputWorkspace='container',   OutputWorkspace='container_single', MaskingInformation=mask_info)
    #MaskBinsFromTable(InputWorkspace='sample',      OutputWorkspace='sample_raw_single',MaskingInformation=mask_info)

    #-----------------------------------------------------------------------------------------#
    # Get sinlge, merged spectrum from banks

    CloneWorkspace(InputWorkspace=sam_corrected, OutputWorkspace='sam_single')
    CloneWorkspace(InputWorkspace=van_corrected, OutputWorkspace='van_single')
    CloneWorkspace(InputWorkspace='container', OutputWorkspace='container_single')
    CloneWorkspace(InputWorkspace='sample', OutputWorkspace='sample_raw_single')
    CloneWorkspace(InputWorkspace='background', OutputWorkspace='background_single')

    SumSpectra(InputWorkspace='sam_single', OutputWorkspace='sam_single',
               ListOfWorkspaceIndices=wkspIndices)
    SumSpectra(InputWorkspace='van_single', OutputWorkspace='van_single',
               ListOfWorkspaceIndices=wkspIndices)

    # Diagnostic workspaces
    SumSpectra(InputWorkspace='container_single', OutputWorkspace='container_single',
               ListOfWorkspaceIndices=wkspIndices)
    SumSpectra(InputWorkspace='sample_raw_single', OutputWorkspace='sample_raw_single',
               ListOfWorkspaceIndices=wkspIndices)
    SumSpectra(InputWorkspace='background_single', OutputWorkspace='background_single',
               ListOfWorkspaceIndices=wkspIndices)

    #-----------------------------------------------------------------------------------------#
    # Merged S(Q) and F(Q)

    # do the division correctly and subtract off the material specific term
    CloneWorkspace(InputWorkspace='sam_single', OutputWorkspace='SQ_ws')
    SQ = (1./bcoh_avg_sqrd)*mtd['SQ_ws'] - (term_to_subtract-1.)  # +1 to get back to S(Q)

    CloneWorkspace(InputWorkspace='sam_single', OutputWorkspace='FQ_ws')
    FQ_raw = mtd['FQ_ws']
    FQ = FQ_raw - self_scat

    qmax = 48.0
    Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
        StartX=high_q_linear_fit_range*qmax, EndX=qmax, # range cannot include area with NAN
        InputWorkspace='SQ', Output='SQ', OutputCompositeMembers=True)
    fitParams = mtd['SQ_Parameters']

    qmax = getQmaxFromData('FQ', WorkspaceIndex=0)
    Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
        StartX=high_q_linear_fit_range*qmax, EndX=qmax, # range cannot include area with NAN
        InputWorkspace='FQ', Output='FQ', OutputCompositeMembers=True)
    fitParams = mtd['FQ_Parameters']

    qmax = 48.0
    Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
        StartX=high_q_linear_fit_range*qmax, EndX=qmax, # range cannot include area with NAN
        InputWorkspace='FQ_raw', Output='FQ_raw', OutputCompositeMembers=True)
    fitParams = mtd['FQ_raw_Parameters']

    # Save dat file
    header_lines = ['<b^2> : %f ' % btot_sqrd_avg, \
                    '<b>^2 : %f ' % bcoh_avg_sqrd, \
                    'self scattering: %f ' % self_scat, \
                    'fitrange: %f %f '  % (high_q_linear_fit_range*qmax,qmax), \
                    'for merged banks %s: %f + %f * Q' % (','.join([ str(i) for i in wkspIndices]), \
                                                       fitParams.cell('Value', 0), fitParams.cell('Value', 1)) ]


    save_file(mtd['sample_raw_single'], title+'_merged_sample_raw.dat',        header=header_lines)
    save_file(mtd['container_single'],  title+'_merged_container.dat',         header=header_lines)
    save_file(mtd['sam_single'],        title+'_merged_sample_minus_background.dat', header=header_lines)
    save_file(mtd['van_single'],        title+'_merged_vanadium.dat',          header=header_lines)
    save_file(mtd['background_single'], title+'_merged_background.dat',          header=header_lines)
    save_file(SQ,                       title+'_merged_sample_normalized.dat', header=header_lines)

    save_file(FQ,                       title+'_FQ.dat', header=header_lines)
    save_file(FQ_raw,                   title+'_FQ_raw.dat', header=header_lines)

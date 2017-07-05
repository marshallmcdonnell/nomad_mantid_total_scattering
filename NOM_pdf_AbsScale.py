#!/usr/bin/env python
import os
import copy
import glob
import re
import argparse
import ConfigParser
from h5py import File

import json
from mantid import mtd
from mantid.simpleapi import *
import numpy as np
import sys
import fileinput



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

def GenerateEventsFilterFromFiles(filenames, OutputWorkspace,
                                  InformationWorkspace, **kwargs):
    pass

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

#-----------------------------------------------------------------------------------
# . NOM_pdf
configfile = sys.argv[1]
print "loading config from", configfile
with open(configfile) as handle:
    config = json.loads(handle.read())

mode = str(config.get('mode',None))

title = str(config['title'])
sam_scans = config['sam']
can = config['can']
van_scans = config['van']
van_bg = config['van_bg']
if mode != 'check_levels':
    material = str(config['material'])
calib = str(config['calib'])
charac = str(config['charac'])
binning= config['binning']
high_q_linear_fit_range = config['high_q_linear_fit_range']
wkspIndices=config['sumbanks'] # workspace indices - zero indexed arrays
packing_fraction = config.get('packing_fraction',None)
cache_dir = config.get("CacheDir", os.path.abspath('.') )

sam = ','.join(['NOM_%d' % num for num in sam_scans])
can = ','.join(['NOM_%d' % num for num in can])
van = ','.join(['NOM_%d' % num for num in van_scans])
van_bg = ','.join(['NOM_%d' % num for num in van_bg])


'''
print "#-----------------------------------#"
print "# BaTiO3 test"
print "#-----------------------------------#"
getAbsScaleInfoFromNexus([78223],ChemicalFormula="Ba1 Ti1 O3", PackingFraction=0.6,SampleMassDensity=2.95,Geometry={"Radius" : 0.29})
'''

print "#-----------------------------------#"
print "# Sample"
print "#-----------------------------------#"
natoms, self_scat = getAbsScaleInfoFromNexus(sam_scans,PackingFraction=packing_fraction,Geometry={"Radius" : 0.3, "Height" : 6.0}, ChemicalFormula=material)

print "#-----------------------------------#"
print "# Vanadium"
print "#-----------------------------------#"
diaV = 0.585
nvan_atoms, tmp = getAbsScaleInfoFromNexus(van_scans,PackingFraction=1.0,SampleMassDensity=6.11,Geometry={"Radius" : diaV/2.0, "Height" : 6.0},ChemicalFormula="V")

print "Sample natoms:", natoms
print "Vanadium natoms:", nvan_atoms
print "Vanadium natoms / Sample natoms:", nvan_atoms/natoms

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
alignAndFocusArgs['PreserveEvents'] = True
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
#alignAndFocusArgs['CacheDir'] = '/tmp' # TODO calculate this and set permissions

##########
# sample
# container
# vanadium
# vanadium_background
# sam_corrected
# van_corrected
# SQ_banks
# sam_single
# van_single
# SQ
##########


#Load(Filename='/home/pf9/Dropbox/AdvancedDiffractionGroup/NOMADsoftware/scripts/marshall/absorption_V_0_58.nxs',
#     OutputWorkspace='van_absorption')

AlignAndFocusPowderFromFiles(OutputWorkspace='sample', Filename=sam, Absorption=None, **alignAndFocusArgs)
NormaliseByCurrent(InputWorkspace='sample', OutputWorkspace='sample',
                   RecalculatePCharge=True)
SaveNexusProcessed(mtd['sample'], os.path.abspath('.') + '/sample_nexus.nxs')

PDDetermineCharacterizations(InputWorkspace='sample',
                             Characterizations='characterizations',
                             ReductionProperties='__snspowderreduction')
propMan = PropertyManagerDataService.retrieve('__snspowderreduction')
qmax = 2.*np.pi/propMan['d_min'].value
qmin = 2.*np.pi/propMan['d_max'].value
for a,b in zip(qmin, qmax):
    print 'Qrange:', a, b
mask_info = generateCropingTable(qmin, qmax)

AlignAndFocusPowderFromFiles(OutputWorkspace='container', Filename=can, Absorption=None, **alignAndFocusArgs)
can = 'container'
NormaliseByCurrent(InputWorkspace=can, OutputWorkspace=can,
                   RecalculatePCharge=True)
SaveNexusProcessed(mtd['container'], os.path.abspath('.') + '/container_nexus.nxs')

AlignAndFocusPowderFromFiles(OutputWorkspace='vanadium', Filename=van, Absorption=None, **alignAndFocusArgs)
van = 'vanadium'
NormaliseByCurrent(InputWorkspace=van, OutputWorkspace=van,
                   RecalculatePCharge=True)
SaveNexusProcessed(mtd['vanadium'], os.path.abspath('.') + '/vanadium_nexus.nxs')

AlignAndFocusPowderFromFiles(OutputWorkspace='vanadium_background', Filename=van_bg, Absorption=None, **alignAndFocusArgs)
van_bg = 'vanadium_background'
NormaliseByCurrent(InputWorkspace=van_bg, OutputWorkspace=van_bg,
                   RecalculatePCharge=True)
SaveNexusProcessed(mtd['vanadium_background'], os.path.abspath('.') + '/vanadium_background_nexus.nxs')

# Multiple-Scattering for Vanadium
SetSampleMaterial(InputWorkspace='vanadium', ChemicalFormula='V')
ConvertUnits(InputWorkspace='vanadium', OutputWorkspace='vanadium', Target="Wavelength", EMode="Elastic")
MultipleScatteringCylinderAbsorption(InputWorkspace='vanadium', OutputWorkspace='vanadium', CylinderSampleRadius=diaV/2.0)




for name in ['sample', can, van, van_bg]:
    ConvertUnits(InputWorkspace=name, OutputWorkspace=name,
                 Target='MomentumTransfer', EMode='Elastic')

sam_corrected = 'sam_corrected'
van_corrected = 'van_corrected'
Minus(LHSWorkspace='sample', RHSWorkspace=can, OutputWorkspace=sam_corrected)
Minus(LHSWorkspace=van, RHSWorkspace=van_bg, OutputWorkspace=van_corrected)
if alignAndFocusArgs['PreserveEvents']:
    CompressEvents(InputWorkspace=sam_corrected, OutputWorkspace=sam_corrected)
    CompressEvents(InputWorkspace=van_corrected, OutputWorkspace=van_corrected)

if mode != 'check_levels':
    SetSampleMaterial(InputWorkspace=sam_corrected, ChemicalFormula=material)
SetSampleMaterial(InputWorkspace=van_corrected, ChemicalFormula='V')

ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
             Target='dSpacing', EMode='Elastic')
StripVanadiumPeaks(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                   BackgroundType='Quadratic')
ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
             Target='TOF', EMode='Elastic')
FFTSmooth(InputWorkspace=van_corrected,
          OutputWorkspace=van_corrected,
          Filter="Butterworth",
          Params='20,2',
          IgnoreXBins=True,
          AllSpectra=True)
ConvertUnits(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
             Target='MomentumTransfer', EMode='Elastic')
SetUncertainties(InputWorkspace=van_corrected, OutputWorkspace=van_corrected,
                 SetError='zero')

material = mtd[sam_corrected].sample().getMaterial()
bcoh_avg_sqrd = material.cohScatterLength()*material.cohScatterLength()
btot_sqrd_avg = material.totalScatterLengthSqrd()
term_to_subtract = btot_sqrd_avg / bcoh_avg_sqrd

for bank in range(mtd['van_corrected'].getNumberHistograms()):
    x_data = mtd['van_corrected'].readX(bank)[0:-1]
    y_data = mtd['van_corrected'].readY(bank)
    bank_title='vanadium_bank_'+str(bank)+'.dat'
    with open(bank_title,'a') as f:
        for x, y in zip(x_data, y_data):
            f.write("%f %f \n" % (x, y))

SQ_banks =  (1./bcoh_avg_sqrd)*mtd['sam_corrected']/mtd['van_corrected'] - (term_to_subtract-1.) 

##################################################################################################
# F(Q) section

print 'self_scat:', self_scat

sigma_v = mtd['van_corrected'].sample().getMaterial().totalScatterXSection()
prefactor = (nvan_atoms/natoms) * ( sigma_v / (4.*np.pi) )
print "Prefactor term:", prefactor
FQ_banks_raw = prefactor * mtd['sam_corrected'] / mtd['van_corrected'] 
FQ_banks = FQ_banks_raw - self_scat 

##################################################################################################

def getQmaxFromData(Workspace=None, WorkspaceIndex=0):
    if Workspace is None:
        return None
    return max(mtd[Workspace].readX(WorkspaceIndex))

# fit the last 80% of the bank being used
for i, q in zip(range(mtd['sam_corrected'].getNumberHistograms()), qmax):
    qmax_data = getQmaxFromData('sam_corrected', i)
    qmax[i] = q if q <= qmax_data else qmax_data

fitrange_individual = [(high_q_linear_fit_range*q, q) for q in qmax]

for q in qmax:
    print 'Linear Fit Qrange:', high_q_linear_fit_range*q, q

for i, fitrange in enumerate(fitrange_individual):
    print 'fitrange:', fitrange[0], fitrange[1]

    Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
        WorkspaceIndex=i,
        StartX=fitrange[0], EndX=fitrange[1], # range cannot include area with NAN
        InputWorkspace='SQ_banks', Output='SQ_banks', OutputCompositeMembers=True)
    fitParams = mtd['SQ_banks_Parameters']

    bank_title=title+'_bank_'+str(i)+'.dat'
    with open(bank_title,'w') as f:
        f.write('#<b^2> : %f \n' % btot_sqrd_avg)
        f.write('#<b>^2 : %f \n' % bcoh_avg_sqrd)
        f.write('#fitrange: %f %f \n' % (fitrange[0], fitrange[1]))
        f.write('#for bank%d: %f + %f * Q\n' % (i+1, fitParams.cell('Value', 0), fitParams.cell('Value', 1)))

for i, fitrange in enumerate(fitrange_individual):
    print 'fitrange:', fitrange[0], fitrange[1]

    Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
        WorkspaceIndex=i,
        StartX=fitrange[0], EndX=fitrange[1], # range cannot include area with NAN
        InputWorkspace='FQ_banks', Output='FQ_banks', OutputCompositeMembers=True)
    fitParams = mtd['FQ_banks_Parameters']

    bank_title=title+'_bank_'+str(i)+'.dat'
    with open(bank_title,'w') as f:
        f.write('#fitrange: %f %f \n' % (fitrange[0], fitrange[1]))
        f.write('#for bank%d: %f + %f * Q\n' % (i+1, fitParams.cell('Value', 0), fitParams.cell('Value', 1)))

for i, fitrange in enumerate(fitrange_individual):
    print 'fitrange:', fitrange[0], fitrange[1]
    Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
        WorkspaceIndex=i,
        StartX=fitrange[0], EndX=fitrange[1], # range cannot include area with NAN
        InputWorkspace='FQ_banks_raw', Output='FQ_banks_raw', OutputCompositeMembers=True)
    fitParams = mtd['FQ_banks_raw_Parameters']

    bank_title=title+'_bank_'+str(i)+'.dat'
    with open(bank_title,'w') as f:
        f.write('#fitrange: %f %f \n' % (fitrange[0], fitrange[1]))
        f.write('#for bank%d: %f + %f * Q\n' % (i+1, fitParams.cell('Value', 0), fitParams.cell('Value', 1)))

for bank in range(SQ_banks.getNumberHistograms()):
    x_data = SQ_banks.readX(bank)[0:-1]
    y_data = SQ_banks.readY(bank)
    bank_title=title+'_bank_'+str(bank)+'.dat'
    with open(bank_title,'a') as f:
        for x, y in zip(x_data, y_data):
            f.write("%f %f \n" % (x, y))

for bank in range(FQ_banks.getNumberHistograms()):
    x_data = FQ_banks.readX(bank)[0:-1]
    y_data = FQ_banks.readY(bank)
    bank_title=title+'_bank_'+str(bank)+'.dat'
    with open(bank_title,'a') as f:
        for x, y in zip(x_data, y_data):
            f.write("%f %f \n" % (x, y))
for bank in range(FQ_banks_raw.getNumberHistograms()):
    x_data = FQ_banks_raw.readX(bank)[0:-1]
    y_data = FQ_banks_raw.readY(bank)
    bank_title=title+'_bank_'+str(bank)+'.dat'
    with open(bank_title,'a') as f:
        for x, y in zip(x_data, y_data):
            f.write("%f %f \n" % (x, y))

Rebin(InputWorkspace=sam_corrected, OutputWorkspace=sam_corrected, Params=binning, PreserveEvents=False)
Rebin(InputWorkspace=van_corrected, OutputWorkspace=van_corrected, Params=binning, PreserveEvents=False)
Rebin(InputWorkspace='container',   OutputWorkspace='container',   Params=binning, PreserveEvents=False)
Rebin(InputWorkspace='sample',      OutputWorkspace='sample',      Params=binning, PreserveEvents=False)
Rebin(InputWorkspace=van_bg,        OutputWorkspace='background',      Params=binning, PreserveEvents=False)

#MaskBinsFromTable(InputWorkspace=sam_corrected, OutputWorkspace='sam_single',       MaskingInformation=mask_info)
#MaskBinsFromTable(InputWorkspace=van_corrected, OutputWorkspace='van_single',       MaskingInformation=mask_info)
#MaskBinsFromTable(InputWorkspace='container',   OutputWorkspace='container_single', MaskingInformation=mask_info)
#MaskBinsFromTable(InputWorkspace='sample',      OutputWorkspace='sample_raw_single',MaskingInformation=mask_info)

RenameWorkspace(InputWorkspace=sam_corrected, OutputWorkspace='sam_single')
RenameWorkspace(InputWorkspace=van_corrected, OutputWorkspace='van_single')
RenameWorkspace(InputWorkspace='container', OutputWorkspace='container_single')
RenameWorkspace(InputWorkspace='sample', OutputWorkspace='sample_raw_single')
RenameWorkspace(InputWorkspace='background', OutputWorkspace='background_single')

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

# do the division correctly and subtract off the material specific term
SQ = (1./bcoh_avg_sqrd)*mtd['sam_single']/mtd['van_single'] - (term_to_subtract-1.)  # +1 to get back to S(Q)

##################################################################################################
# F(Q) section
FQ_raw = prefactor * mtd['sam_single']/mtd['van_single']
FQ = FQ_raw - self_scat
##################################################################################################

'''
Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
    StartX=high_q_linear_fit_range*qmax, EndX=qmax, # range cannot include area with NAN
    InputWorkspace='SQ', Output='SQ', OutputCompositeMembers=True)
fitParams = mtd['SQ_Parameters']

qmax = getQmaxFromData('FQ', WorkspaceIndex=0)
Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
    StartX=high_q_linear_fit_range*qmax, EndX=qmax, # range cannot include area with NAN
    InputWorkspace='FQ', Output='FQ', OutputCompositeMembers=True)
fitParams = mtd['FQ_Parameters']
'''

qmax = 48.0
Fit(Function='name=LinearBackground,A0=1.0,A1=0.0',
    StartX=high_q_linear_fit_range*qmax, EndX=qmax, # range cannot include area with NAN
    InputWorkspace='FQ_raw', Output='FQ_raw', OutputCompositeMembers=True)
fitParams = mtd['FQ_raw_Parameters']

# Save dat file
header_lines = ['<b^2> : %f ' % btot_sqrd_avg, \
                '<b>^2 : %f ' % bcoh_avg_sqrd, \
                'fitrange: %f %f '  % (high_q_linear_fit_range*qmax,qmax), \
                'for merged banks %s: %f + %f * Q' % (','.join([ str(i) for i in wkspIndices]), \
                                                   fitParams.cell('Value', 0), fitParams.cell('Value', 1)) ]

sample_by_vanadium = mtd['sample_raw_single'] / mtd['van_single']
container_by_vanadium = mtd['container_single'] / mtd['van_single']
background_by_vanadium = mtd['background_single'] / mtd['van_single']

save_file(mtd['sample_raw_single'], title+'_merged_sample_raw.dat',        header=header_lines)
save_file(mtd['container_single'],  title+'_merged_container.dat',         header=header_lines)
save_file(mtd['sam_single'],        title+'_merged_sample_minus_background.dat', header=header_lines)
save_file(mtd['van_single'],        title+'_merged_vanadium.dat',          header=header_lines)
save_file(mtd['background_single'], title+'_merged_background.dat',          header=header_lines)
save_file(SQ,                       title+'_merged_sample_normalized.dat', header=header_lines)
save_file(sample_by_vanadium,       title+'_merged_sample_by_vanadium.dat', header=header_lines)
save_file(container_by_vanadium,    title+'_merged_container_by_vanadium.dat', header=header_lines)
save_file(background_by_vanadium,   title+'_merged_background_by_vanadium.dat',   header=header_lines)

save_file(FQ,                       title+'_FQ.dat', header=header_lines)
save_file(FQ_raw,                   title+'_FQ_raw.dat', header=header_lines)

#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import json
from mantid.simpleapi import *

#-----------------------------------------------------------------------------------
# Load file

configfile = sys.argv[1]
print "loading config from", configfile
with open(configfile) as handle:
    config = json.loads(handle.read())

sam_scans = config['run']
calib = str(config['calib'])
charac = str(config['charac'])

sam_scans = ','.join(['NOM_%s' % sam for sam in sam_scans])

#-----------------------------------------------------------------------------------
# Setup arguments

results = PDLoadCharacterizations(Filename=charac, OutputWorkspace='characterizations')
alignArgs = dict(PrimaryFlightPath = results[2],
                         SpectrumIDs       = results[3],
                         L2                = results[4],
                         Polar             = results[5],
                         Azimuthal         = results[6])

alignArgs['CalFilename'] = calib
#alignArgs['ResampleX'] = -6000
alignArgs['RemovePromptPulseWidth'] = 50
alignArgs['Characterizations'] = 'characterizations'
alignArgs['ReductionProperties'] = '__snspowderreduction'


for x, y in alignArgs.iteritems():
    print x, y

#-----------------------------------------------------------------------------------
# Load and Align
wksp='wksp'
Load(Filename=sam_scans, OutputWorkspace=wksp)
PDDetermineCharacterizations(InputWorkspace=wksp,
                             Characterizations=alignArgs['Characterizations'],
                             ReductionProperties=alignArgs['ReductionProperties'])
Load(Filename=sam_scans, OutputWorkspace=wksp)
LoadDiffCal(InputWorkspace=wksp,
            Filename=alignArgs['CalFilename'],
            WorkspaceName='NOM',
            TofMin=300, TofMax=16666)
CompressEvents(InputWorkspace=wksp,OutputWorkspace=wksp, Tolerance=0.01)
CropWorkspace(InputWorkspace=wksp,OutputWorkspace=wksp, XMin=300, XMax=16666)
RemovePromptPulse(InputWorkspace=wksp,OutputWorkspace=wksp, Width=alignArgs['RemovePromptPulseWidth'])
MaskDetectors(Workspace=wksp,MaskedWorkspace='NOM_mask')
AlignDetectors(InputWorkspace=wksp,OutputWorkspace=wksp, CalibrationWorkspace='NOM_cal')
ConvertUnits(InputWorkspace=wksp,OutputWorkspace=wksp,Target='TOF')
ConvertUnits(InputWorkspace=wksp,OutputWorkspace=wksp,Target='Wavelength')
CropWorkspace(InputWorkspace=wksp,OutputWorkspace=wksp, XMin=0.1, XMax=2.9)
ConvertUnits(InputWorkspace=wksp,OutputWorkspace=wksp,Target='dSpacing')
Rebin(InputWorkspace=wksp,OutputWorkspace=wksp,Params='0.0,0.001,6.0')
SaveNexus(InputWorkspace=wksp, Filename=sam_scans+'_aligned.nxs',Title='Aligned Scan '+sam_scans)

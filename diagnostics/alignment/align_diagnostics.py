#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import json
from mantid.simpleapi import *

#-------------------------------------------------------------------------
# Load file

configfile = sys.argv[1]
print("loading config from", configfile)
with open(configfile) as handle:
    config = json.loads(handle.read())

sam_scans = config['Run']
calib = config.get('CalFilename', None)
charac = config.get('CharacterizationFilename', None)
grouping = config.get('GroupingFileName', None)
save_dir = config.get('SaveDirectory', '/tmp')
instrument = config.get('Instrument', 'NOM')
idf = config.get('InstrumentDefinitionFile', None)
tof = config.get('Tof', '300,16666')
if idf is not None:
    idf = str(idf)

tofMin, tofMax = [ float(t) for t in tof.split(',')] 


sam_scans = ','.join(['%s_%s' % (instrument, sam) for sam in sam_scans])

# Override if Filename present
if "Filename" in config:
    sam_scans = config["Filename"]

#-------------------------------------------------------------------------
# Setup arguments

alignArgs = dict()
if charac:
    results = PDLoadCharacterizations(
        Filename=charac,
        OutputWorkspace='characterizations')
    alignArgs = dict(PrimaryFlightPath=results[2],
                     SpectrumIDs=results[3],
                     L2=results[4],
                     Polar=results[5],
                     Azimuthal=results[6])
    alignArgs['Characterizations'] = 'characterizations'
    alignArgs['ReductionProperties'] = '__snspowderreduction'

if calib:
    alignArgs['CalFilename'] = calib

alignArgs['RemovePromptPulseWidth'] = 50
alignArgs['CompressTolerance'] = 0.0001


for x, y in alignArgs.iteritems():
    print(x, y)

#-------------------------------------------------------------------------
# Load and Align
wksp = 'wksp'
Load(Filename=sam_scans, OutputWorkspace=wksp)

# Load in a different Instrument Definition File from one found in NeXus
if idf:
    LoadInstrument(Workspace=wksp,
                   Filename=idf,
                   RewriteSpectraMap=False)

ConvertUnits(InputWorkspace=wksp, OutputWorkspace=wksp, Target='TOF')
print('After Load, we have : {num_histo} histograms'.format(
    num_histo=mtd[wksp].getNumberHistograms()))
if calib:
    LoadDiffCal(InputWorkspace=wksp,
                Filename=alignArgs['CalFilename'],
                WorkspaceName=instrument,
                TofMin=tofMin, TofMax=tofMax)
    print('After LoadDiffCal, we have : {num_histo} histograms'.format(
        num_histo=mtd[wksp].getNumberHistograms()))
CompressEvents(InputWorkspace=wksp, OutputWorkspace=wksp,
               Tolerance=alignArgs['CompressTolerance'])
CropWorkspace(InputWorkspace=wksp, OutputWorkspace=wksp, XMin=tofMin, XMax=tofMax)
RemovePromptPulse(InputWorkspace=wksp, OutputWorkspace=wksp,
                  Width=alignArgs['RemovePromptPulseWidth'])
if calib:
    mask = '%s_mask' % instrument
    cal = '%s_cal' % instrument
    # MaskDetectors(Workspace=wksp,MaskedWorkspace=mask)
    AlignDetectors(
        InputWorkspace=wksp,
        OutputWorkspace=wksp,
        CalibrationWorkspace=cal)
    print('After Mask and Align, we have : {num_histo} histograms'.format(
        num_histo=mtd[wksp].getNumberHistograms()))
CompressEvents(InputWorkspace=wksp, OutputWorkspace=wksp,
               Tolerance=alignArgs['CompressTolerance'])
ConvertUnits(InputWorkspace=wksp, OutputWorkspace=wksp, Target='TOF')
ConvertUnits(InputWorkspace=wksp, OutputWorkspace=wksp, Target='Wavelength')
CropWorkspace(InputWorkspace=wksp, OutputWorkspace=wksp, XMin=0.1, XMax=2.9)
ConvertUnits(InputWorkspace=wksp, OutputWorkspace=wksp, Target='dSpacing')
Rebin(InputWorkspace=wksp, OutputWorkspace=wksp, Params='0.0,0.0001,6.0')

if config["OutputFilename"]:
    filename = config["OutputFilename"]
    filename = os.path.join(save_dir, filename)
else:
    filename = os.path.join(save_dir, sam_scans + '_aligned.nxs')


SaveNexus(
    InputWorkspace=wksp,
    Filename=filename,
    Title='Aligned Scan ' +
    sam_scans)

# Focussing
if grouping:
    LoadDetectorsGroupingFile(InputFile=grouping, OutputWorkspace="group_ws")
    DiffractionFocussing(
        InputWorkspace=wksp,
        OutputWorkspace=wksp,
        GroupingWorkspace="group_ws")
    filename = os.path.join(save_dir, sam_scans + '_aligned_focussed.nxs')
    SaveNexus(
        InputWorkspace=wksp,
        Filename=filename,
        Title='Aligned+Focussed Scan ' +
        sam_scans)

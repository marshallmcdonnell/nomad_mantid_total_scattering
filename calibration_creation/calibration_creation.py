#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import datetime
import sys
import os
import time
import json
import matplotlib.pyplot as plt

from mantid.simpleapi import *

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

infile = os.path.abspath(sys.argv[1])
print('reading \'%s\'' % infile)
with open(infile) as handle:
    args = json.loads(handle.read())

pdcal_defaults = { 'TofBinning' : [300, -.001, 16666.7],
                   'StartFromObservedPeakCentre' : True,
                   'CalibrationParameters' : 'DIFC' }

calibrants = args['Calibrants']
idf = args.get('InstrumentDefinitionFile', None)
oldCal = args.get('OldCal',None)
chunkSize = int(args.get('ChunkSize', 8))
filterBadPulses = int(args.get('FilterBadPulses', 25))
pdcal_kwargs = args.get('PDCalibration', dict())
pdcal_kwargs = merge_two_dicts(pdcal_defaults, pdcal_kwargs)

# General date and cali directory for batch calibrations
date_master = str(args.get('Date', datetime.datetime.now().strftime('%Y_%m_%d')))
caldirectory_master = str(args.get('CalDirectory', os.path.abspath('.')))

# PDCalibration
# -------------------------------------

for calibrant in calibrants:
   
    # Calibrant specific date and directory 
    date = str(calibrants[calibrant].get('Date', date_master))
    caldirectory = str(calibrants[calibrant].get('CalDirectory', caldirectory_master))

    if 'SampleEnvironment' in calibrants[calibrant]:
        samp_env = str(calibrants[calibrant]['SampleEnvironment'])
    else:
        samp_env = str(args['SampleEnvironment'])

    if 'Vanadium' in calibrants[calibrant]:
        vanadium_args = calibrants[calibrant]['Vanadium']
        if "Filename" in vanadium_args:
            vanadium = vanadium_args["Filename"]
        else:
            vanadium = 'NOM_%d' % int(vanadium_args["RunNumber"])
    else:
        vanadium = 0

    # Create calibration output filename
    runNumber = int(calibrant)
    calfilename = caldirectory + \
        '/NOM_d%d_%s_%s.h5' % (runNumber, date, samp_env)
    print('going to create calibration file: %s' % calfilename)

    # Specify inpute event filename
    filename = 'NOM_%d' % runNumber
    wkspName = 'NOM_%d' % runNumber
    if 'Filename' in calibrants[calibrant]:
        filename = calibrants[calibrant]['Filename']


    # Way to check for file if it doesn't exist just using Filename
    def FileExists(filename):
        try:
            Load(
                Filename=filename,
                OutputWorkspace='tmp',
                MaxChunkSize=chunkSize,
                MetaDataOnly=True)
            DeleteWorkspace('tmp')
            return True
        except ValueError:
            return False

    print("Waiting for calibration NeXus file...")
    while not FileExists(filename):
        time.sleep(30)
    print("Found calibration NeXus file!")

    # Load workspace
    Load(Filename=filename,
         OutputWorkspace=wkspName,
         MaxChunkSize=chunkSize,
         FilterBadPulses=filterBadPulses)
    CropWorkspace(
        InputWorkspace=wkspName,
        OutputWorkspace=wkspName,
        XMin=300,
        XMax=16666.7)

    # Load in a different Instrument Definition File from one found in NeXus
    if idf:
        print('HERE',idf)
        LoadInstrument(Workspace=wkspName,
                       Filename=str(idf),
                       RewriteSpectraMap=False)

    # NOMAD uses tabulated reflections for diamond
    dvalues = (0.3117, 0.3257, 0.3499, 0.4205, 0.4645, 0.4768, 0.4996, 0.5150, 0.5441,
               0.5642, 0.5947, 0.6307, .6866, .7283, .8185, .8920, 1.0758, 1.2615, 2.0599)

    if oldCal is not None:
        pdcal_kwargs['PreviousCalibration'] = oldCal

    print(pdcal_kwargs)
    PDCalibration(SignalWorkspace=wkspName,
                  PeakPositions=dvalues,
                  OutputCalibrationTable='new_cal',
                  DiagnosticWorkspaces='diagnostics',
                  **pdcal_kwargs)

    dbinning = (.01, -.001, 3.)
    AlignDetectors(
        InputWorkspace=wkspName,
        OutputWorkspace=wkspName,
        CalibrationWorkspace='new_cal')
    CropWorkspace(InputWorkspace=wkspName, OutputWorkspace=wkspName,
                  XMin=dbinning[0], XMax=dbinning[2])
    Rebin(InputWorkspace=wkspName, OutputWorkspace=wkspName, Params=dbinning)

    # [DetectorDiagnostic](http://docs.mantidproject.org/nightly/algorithms/DetectorDiagnostic-v1.html) uses statistical criteria for determining what pixels should be used to produce final data

    DeleteWorkspace(wkspName)

    if vanadium > 0:
        Load(
            Filename=vanadium,
            OutputWorkspace=vanadium,
            MaxChunkSize=chunkSize,
            FilterBadPulses=filterBadPulses)
        DetectorDiagnostic(InputWorkspace=vanadium, OutputWorkspace='NOM_mask_detdiag',
                           RangeLower=300, RangeUpper=16666.7,  # TOF range to use
                           LowThreshold=10,  # minimum number of counts for a detector
                           LevelsUp=1)  # median calculated from the tube
        DeleteWorkspace(vanadium)

        # The result of `DetectorDiagnostic` can be combined with the result of
        # the mask generated by `PDCalibration` using
        # [BinaryOperateMasks](http://docs.mantidproject.org/nightly/algorithms/BinaryOperateMasks-v1.html)
        BinaryOperateMasks(
            InputWorkspace1='new_cal_mask',
            InputWorkspace2='NOM_mask_detdiag',
            OperationType='OR',
            OutputWorkspace='NOM_mask_final')
    else:
        RenameWorkspace(
            InputWorkspace='new_cal_mask',
            OutputWorkspace='NOM_mask_final')

    # The only information missing for `SaveDiffCal` is which pixels to
    # combine to make an output spectrum. This is done using
    # [CreateGroupingWorkspace](http://docs.mantidproject.org/nightly/algorithms/CreateGroupingWorkspace-v1.html).
    # For NOMAD, the `Column` option will generate 6 spectra. An alternative
    # is to generate a grouping file to load with
    # [LoadDetectorsGroupingFile](http://docs.mantidproject.org/nightly/algorithms/LoadDetectorsGroupingFile-v1.html).
    CreateGroupingWorkspace(InstrumentName='NOMAD', GroupDetectorsBy='Group',
                            OutputWorkspace='NOM_group')

    print('saving file', calfilename)
    SaveDiffCal(CalibrationWorkspace='new_cal',
                GroupingWorkspace='NOM_group',
                MaskWorkspace='NOM_mask_final',
                Filename=calfilename)

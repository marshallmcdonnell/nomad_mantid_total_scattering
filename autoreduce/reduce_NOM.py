#!/usr/bin/env python

import os
import sys
import shutil
sys.path.append("/opt/mantidnightly/bin")
from mantid.simpleapi import *
import mantid

import json

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


configfile = sys.argv[1]
print "loading config from", configfile
with open(configfile) as handle:
    config = json_loads_byteified(handle.read())

title = config.get('title',None)
# Get sample info
sample = config['sam']
samRun = sample['Runs'][0]
sampleBackRun=sample['Background']['Runs'][0]

# Get normalization info
van = config['van']
vanRun=van['Runs'][0]
vanBackRun=van['Background']['Runs'][0]
vanRadius=van['Geometry']['Radius']

# Get calibration, characterization, and other settings
calFile= config['calib']
charFile = config['charac']
cacheDir = config.get("CacheDir", os.path.abspath('.'))
outputDir=config.get("OutputDir", os.path.abspath('.'))

########## user defined parameters
resamplex=-6000
#wavelengthMin=0.1
#wavelengthMax=2.9
# specify files to be summed as a tuple or list
########## end of user defined parameters

maxChunkSize=8.

# setup for MPI
if AlgorithmFactory.exists('GatherWorkspaces'):
     from mpi4py import MPI
     mpiRank = MPI.COMM_WORLD.Get_rank()
else:
     mpiRank = 0

# uncomment next line to delete cache files
#if mpiRank == 0: CleanFileCache(CacheDir=cacheDir, AgeInDays=0)

# determine information for caching
eventFile = 'NOM_%s' % samRun
wksp=LoadEventNexus(Filename=eventFile, MetaDataOnly=True)
PDLoadCharacterizations(Filename=charFile,
                        OutputWorkspace="characterizations")
PDDetermineCharacterizations(InputWorkspace=wksp,
                             Characterizations="characterizations",
                             BackRun=sampleBackRun,
                             NormRun=vanRun,
                             NormBackRun=vanBackRun)
DeleteWorkspace(str(wksp))
DeleteWorkspace("characterizations")
charPM = mantid.PropertyManagerDataService.retrieve('__pd_reduction_properties')

# get back the runs to use so they can be explicit in the generated python script
sampleBackRun = charPM['container'].value[0]
vanRun        = charPM['vanadium'].value[0]
vanBackRun    = charPM['vanadium_background'].value[0]

# work on container cache file
if sampleBackRun > 0:
    canWkspName="NOM_"+str(sampleBackRun)
    canProcessingProperties = ['container', 'd_min', 'd_max',
                               'tof_min', 'tof_max']
    canProcessingOtherProperties = ["ResampleX="+str(resamplex),
                                    "BackgroundSmoothParams="+str(''),
                                    "CalibrationFile="+calFile]

    (canCacheName, _) = CreateCacheFilename(Prefix=canWkspName, CacheDir=cacheDir,
                                            PropertyManager='__pd_reduction_properties',
                                            Properties=canProcessingProperties,
                                            OtherProperties=canProcessingOtherProperties)
    print "Container cache file:", canCacheName

    if os.path.exists(canCacheName):
        print "Loading container cache file '%s'" % canCacheName
        Load(Filename=canCacheName, OutputWorkspace=canWkspName)

# work on vanadium cache file
if vanRun > 0:
    vanWkspName="NOM_"+str(vanRun)
    vanProcessingProperties = ['vanadium', 'vanadium_background', 'd_min', 'd_max',
                               'tof_min', 'tof_max']
    vanProcessingOtherProperties = ["ResampleX="+str(resamplex),
                                    "VanadiumRadius="+str(vanRadius),
                                    "CalibrationFile="+calFile]

    (vanCacheName, _) =  CreateCacheFilename(Prefix=vanWkspName, CacheDir=cacheDir,
                                             PropertyManager='__pd_reduction_properties',
                                             Properties=vanProcessingProperties,
                                             OtherProperties=vanProcessingOtherProperties)
    print "Vanadium cache file:", vanCacheName

    if os.path.exists(vanCacheName):
        print "Loading vanadium cache file '%s'" % vanCacheName
        Load(Filename=vanCacheName, OutputWorkspace=vanWkspName)

# process the run
SNSPowderReduction(Filename=eventFile,
                   MaxChunkSize=maxChunkSize, PreserveEvents=True,PushDataPositive='ResetToZero',
                   CalibrationFile=calFile,
                   CharacterizationRunsFile=charFile,
                   BackgroundNumber=str(sampleBackRun),
                   VanadiumNumber=str(vanRun),
                   VanadiumBackgroundNumber=str(vanBackRun),
                   RemovePromptPulseWidth=50,
                   ResampleX=resamplex,
                   BinInDspace=True,
                   FilterBadPulses=25.,
                   SaveAs="gsas fullprof topas",
                   OutputFilePrefix=title,
                   OutputDirectory=outputDir,
                   StripVanadiumPeaks=True,
                   VanadiumRadius=vanRadius,
                   NormalizeByCurrent=True, FinalDataUnits="dSpacing")

# only write out thing on control job
if mpiRank == 0:
    # save out the container cache file
    if sampleBackRun > 0 and not os.path.exists(canCacheName):
        ConvertUnits(InputWorkspace=canWkspName, OutputWorkspace=canWkspName, Target="TOF")
        SaveNexusProcessed(InputWorkspace=canWkspName, Filename=canCacheName)

    # save out the vanadium cache file
    if vanRun > 0 and not os.path.exists(vanCacheName):
        ConvertUnits(InputWorkspace=vanWkspName, OutputWorkspace=vanWkspName, Target="TOF")
        SaveNexusProcessed(InputWorkspace=vanWkspName, Filename=vanCacheName)

    wksp_name = "NOM_"+str(samRun)

    # save the processing script
    GeneratePythonScript(InputWorkspace=wksp_name,
                     Filename=os.path.join(outputDir,wksp_name+'.py'))

    ConvertUnits(InputWorkspace=wksp_name, OutputWorkspace=wksp_name, Target="dSpacing")

    # save a picture of the normalized ritveld data
    banklabels = ['bank 1 - 15 deg',
                  'bank 2 - 31 deg',
                  'bank 3 - 67 deg',
                  'bank 4 - 122 deg',
                  'bank 5 - 154 deg',
                  'bank 6 - 7 deg']
    spectratoshow = [2,3,4,5]

    saveplot1d_args = dict(InputWorkspace=wksp_name,
                           SpectraList=spectratoshow,
                           SpectraNames=banklabels)

    post_image = True
    if post_image:
        div = SavePlot1D(OutputType='plotly', **saveplot1d_args)
        from postprocessing.publish_plot import publish_plot
        request = publish_plot('NOM', samRun, files={'file':div})
        print "post returned %d" % request.status_code
        print "resulting document:"
        print request.text
    else:
        filename = os.path.join(outputDir, wksp_name + '.html')
        SavePlot1D(OutputFilename=filename, OutputType='plotly-full',
                   **saveplot1d_args)
        print 'saved', filename

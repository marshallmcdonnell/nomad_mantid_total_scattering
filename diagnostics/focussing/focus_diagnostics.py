#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import json
from mantid.simpleapi import *

#-----------------------------------------------------------------------------------
# Load file

configfile = sys.argv[1]
print("loading config from", configfile)
with open(configfile) as handle:
    config = json.loads(handle.read())

sam_scans = config.get('Run', None)
aligned = config['AlignedFilename']
grouping = config['GroupingFileName']

sam_scans = ','.join(['NOM_%s' % sam for sam in sam_scans])

# Load aligned workspace
wksp='wksp'
Load(Filename=aligned, OutputWorkspace=wksp)

# Get Grouping
LoadDetectorsGroupingFile(InputFile=grouping, OutputWorkspace="group_ws")

# Focussing using the grouping
print ('Before Focussing, we have : {num_histo} histograms'.format(num_histo=mtd[wksp].getNumberHistograms()))
DiffractionFocussing(InputWorkspace=wksp, OutputWorkspace=wksp, GroupingWorkspace="group_ws")
print ('After Focussing, we have : {num_histo} histograms'.format(num_histo=mtd[wksp].getNumberHistograms()))
SaveNexus(InputWorkspace=wksp, Filename=sam_scans+'_aligned_focussed.nxs',Title='Aligned+Focussed Scan '+sam_scans)

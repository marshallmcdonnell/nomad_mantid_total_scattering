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

wksp='wksp'
Load(Filename=aligned, OutputWorkspace=wksp)

# Focussing
print( wksp, grouping)
DiffractionFocussing(InputWorkspace=wksp, OutputWorkspace=wksp, GroupingFileName=str(grouping))
SaveNexus(InputWorkspace=wksp, Filename=sam_scans+'_aligned_focussed.nxs',Title='Aligned+Focussed Scan '+sam_scans)

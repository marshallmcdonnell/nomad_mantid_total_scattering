#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import os
import json
from mantid.simpleapi import *

#-------------------------------------------------------------------------
# Load file

configfile = sys.argv[1]
print("loading config from", configfile)
with open(configfile) as handle:
    config = json.loads(handle.read())

aligned = config['AlignedFilename']
grouping = config['GroupingFileName']
save_dir = config.get('SaveDirectory', '/tmp')

# Load aligned workspace
wksp = 'wksp'
Load(Filename=aligned, OutputWorkspace=wksp)

# Get Grouping
LoadDetectorsGroupingFile(InputFile=grouping, OutputWorkspace="group_ws")

# Focussing using the grouping
print('Before Focussing, we have : {num_histo} histograms'.format(
    num_histo=mtd[wksp].getNumberHistograms()))
DiffractionFocussing(
    InputWorkspace=wksp,
    OutputWorkspace=wksp,
    GroupingWorkspace="group_ws")
print('After Focussing, we have : {num_histo} histograms'.format(
    num_histo=mtd[wksp].getNumberHistograms()))
if config["OutputFilename"]:
    filename = config["OutputFilename"]
    filename = os.path.join(save_dir, filename)
else:
    basename = os.path.basename(aligned)
    filename = os.path.splitext(basename)[0] + '_focussed.nxs'
    filename = os.path.join(save_dir, filename)
SaveNexus(InputWorkspace=wksp, Filename=filename, Title='Aligned+Focussed')
print("Saved File: {}".format(filename))

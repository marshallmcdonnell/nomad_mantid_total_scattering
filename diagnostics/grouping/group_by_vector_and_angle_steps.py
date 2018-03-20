#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import json
import argparse
from mantid.simpleapi import *
from mantid.kernel import V3D, Quat

import numpy as np

from diagnostics import grouping

#-----------------------------------------------------------------------------
# Rotates vector about an axis by an angle

def MyRotateFunction(vector, axis, angle):
    from copy import copy
    q = Quat(angle,axis)
    new_vector = copy(vector)
    q.rotate(new_vector)
    return new_vector

#-----------------------------------------------------------------------------
# Uses vector and series of rotations to get output axis vector

def GenerateAxisVector(Rotations=None,Vector=[0,0,1] ):
    # Get vector to manipulate
    x, y, z = Vector
    axis_vector = V3D(x,y,z)

    # Apply rotations
    for rotation in Rotations:
        x, y, z = rotation["Axis"]
        axis_vector = MyRotateFunction(axis_vector, V3D(x,y,z), rotation["Angle"])

    return axis_vector

#-----------------------------------------------------------------------------
# Generates a single grouping file

def GenerateGroupingFile(InputWorkspace=None,CentralVector=V3D(0,0,1),AngleStep=None,GroupingFilename=None, MaskIDs=None):
    # Grab objects of interest
    detectorInfo = mtd[wksp].detectorInfo()
    instrument = mtd[wksp].getInstrument()
    sample     = instrument.getSample()
    rad2deg = 180. / np.pi

    # Setup array with only the detectors (no monitors)
    num_components = detectorInfo.size()
    components = np.arange(num_components, dtype=int)
    detectors = np.array([ i for i in components if not detectorInfo.isMonitor(i) ])
    detectors = grouping.utils.revalue_array(detectors)

    # Have a paralle array to detectors to say what group it belongs in
    grouper   = np.full_like(detectors, -1) 

    # Mean twoTheta and L2
    twoThetaAverage = dict()
    l2Average = dict()

    # Loop over detectors
    for idx in detectors:
        detector = instrument.getDetector(idx)
        l2 = detector.getDistance(sample)
        tt = detector.getTwoTheta(sample.getPos(), CentralVector) * rad2deg

        where = tt / AngleStep
        grouper[detector.getID()] = where

    mask = grouping.utils.apply_mask(detectors, MaskIDs)
    md, mg, groups = grouping.utils.mask_and_group(detectors, grouper, mask)
    grouping.utils.write_grouping_file(GroupingFilename, groups, instrument="NOMAD")
    



#-----------------------------------------------------------------------------
# Load file

if __name__ == "__main__":
    configfile = sys.argv[1]
    print("loading config from", configfile)
    with open(configfile) as handle:
        args = json.loads(handle.read())

    # Load instrument
    if "RunNumber" in args:
        wksp = "%s_%s" % (args["Instrument"], args["RunNumber"])
        Load(Filename=wksp, OutputWorkspace=wksp)

    else:
        wksp = "%s" % (args["Instrument"])
        LoadEmptyInstrument(InstrumentName=args["Instrument"], OutputWorkspace=wksp)

    # Loop over grouping files
    for group in args["Groupings"]:
        mask_ids = None
        if group["Mask"]:
            mask_ids = grouping.utils.create_id_list(**group["Mask"])

        central_vector = GenerateAxisVector(Vector=group["AxisVector"], 
                                         Rotations=group["Rotations"])    
        GenerateGroupingFile(InputWorkspace=wksp,
                             CentralVector=central_vector,
                             AngleStep=group["AngleStep"],
                             GroupingFilename=group["GroupingFilename"],
                             MaskIDs=mask_ids)
                             

'''
Want to replicate this C++ code with arbitrary vector
-----------------------------------------------------

  std::vector<detid_t>::iterator it;
  for (it = dets.begin(); it != dets.end(); ++it) {
    double tt = instrument->getDetector(*it)->getTwoTheta(
                    instrument->getSample()->getPos(), Kernel::V3D(0, 0, 1)) *
                Geometry::rad2deg;
    double r =
        instrument->getDetector(*it)->getDistance(*(instrument->getSample()));
    size_t where = static_cast<size_t>(tt / step);
    groups.at(where).push_back(*it);
    twoThetaAverage.at(where) += tt;
    rAverage.at(where) += r;
}
'''
'''
run=13655
wksp = 'wksp'
Load(Filename="NOM_%d" % 13655, OutputWorkspace=wksp)

instrument = mtd[wksp].getInstrument()
sample = mtd[wksp].getInstrument().getSample().getPos()
vector = V3D(0,0,1)

rad2deg = 180./np.pi
detID = 1
print(instrument.getDetector(detID).getTwoTheta(sample,vector) * rad2deg)
'''

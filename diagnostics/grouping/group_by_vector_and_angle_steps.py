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
# Get new group id revalued by a cutting plane

def GetGroupIdUsingCutPlane(GroupId=None, Point=None, NormalVector=[0,0,1], BaseVector=[0,0,0]):
    if GroupId is None:
        return None
    if Point is None:
        return GroupId

    # Normalize vectors
    '''
    norm = np.linalg.norm(NormalVector)
    if norm == 0.0:
        norm = np.float64(1.0) 
    NormalVector = NormalVector / norm 
    '''

    # Get a, b, c, d form of plane
    a, b, c = NormalVector
    d = np.dot(-1.*np.asarray(NormalVector), BaseVector)
    x, y, z = Point

    WhichSideOfPlane = np.sign(np.dot( (a,b,c,d), (x,y,z,1.)))
    return GroupId * WhichSideOfPlane

#-----------------------------------------------------------------------------
# Generates a single grouping file using the detectors

def GenerateGroupingFileFromDetectors(InputWorkspace=None,CentralVector=V3D(0,0,1),AngleStep=None,InitialGroupingFilename=None, GroupingFilename=None, MaskIDs=None, UseQvectorForAngle=False, CutGroupingWithPlane=None, AtomIdsForVisualization=False):
    # Grab objects of interest
    detectorInfo = mtd[wksp].detectorInfo()
    instrument = mtd[wksp].getInstrument()
    sample     = instrument.getSample()
    rad2deg = 180. / np.pi

    # Setup array with only the detectors (no monitors)
    num_components = detectorInfo.size()
    components = np.arange(num_components, dtype=int)
    detectors = np.array([ i for i in components if not detectorInfo.isMonitor(int(i)) ])
    detectors = grouping.utils.revalue_array(detectors)

    # Have a paralle array to detectors to say what group it belongs in
    grouper   = np.full_like(detectors, -1) 
    debug_grouper   = np.full_like(detectors, -1) 

    # Loop over detectors
    for idx in detectors:
        detector = instrument.getDetector(idx)
        l2 = detector.getDistance(sample)
        tt = detector.getTwoTheta(sample.getPos(), CentralVector) * rad2deg

        if UseQvectorForAngle:
            k_i = V3D(0,0,1)
            k_f = detector.getPos() / np.linalg.norm(detector.getPos())
            q   = k_f - k_i
            q   = q / np.linalg.norm(q)
            v   = CentralVector / np.linalg.norm(CentralVector)
            cos_theta = np.dot(q,v)
            # cos_theta = np.sqrt(cos_theta * cos_theta)
            tt = np.arccos(np.clip( cos_theta, -1, 1)) * rad2deg

        where = int(tt / AngleStep) + 1
        debug_grouper[detector.getID()] = where
        if CutGroupingWithPlane:
            where = GetGroupIdUsingCutPlane(GroupId=where, 
                        Point=detector.getPos(), 
                        NormalVector=CutGroupingWithPlane["NormalVector"], 
                        BaseVector=CutGroupingWithPlane["BaseVector"])
        grouper[detector.getID()] = where

    np.set_printoptions(threshold='nan')
    mask = grouping.utils.apply_mask(detectors, MaskIDs)
    md, mg, groups = grouping.utils.mask_and_group(detectors, grouper, mask)
    grouping.utils.write_grouping_file(GroupingFilename, groups, instrument="NOMAD")

    if AtomIdsForVisualization:
        base=os.path.splitext(os.path.basename(GroupingFilename))[0]
        group_of_pixel = grouping.utils.revalue_array(grouper[mask])

        with open("%s.xyz" % base, 'w') as f:
            f.write("{}\n\n".format(len(md)))
            for group_id, pixel_idx in zip(group_of_pixel, md):
                x, y, z = instrument.getDetector(pixel_idx).getPos()
                print_args = {'idx' : pixel_idx, 
                              'x' : x, 
                              'y' : y, 
                              'z' : z, 
                              'group_id' : group_id 
                }

                print_str="{group_id} {x} {y} {z}\n"
                f.write(print_str.format(**print_args))

#-----------------------------------------------------------------------------
# Generates a single grouping file using spectra 

def GenerateGroupingFileFromSpectra(InputWorkspace=None,CentralVector=V3D(0,0,1),AngleStep=None,InitialGroupingFilename=None, GroupingFilename=None, MaskIDs=None):
    # Grab objects of interest
    instrument = mtd[wksp].getInstrument()
    sample     = instrument.getSample()
    rad2deg = 180. / np.pi

    # Group the workspace
    grp_wksp = 'grouping_wksp'
    LoadDetectorsGroupingFile(InputFile=InitialGroupingFilename,OutputWorkspace=grp_wksp)
    GroupDetectors(InputWorkspace=wksp, OutputWorkspace=wksp, CopyGroupingFromWorkspace=grp_wksp)

    # Get spectra list
    spectra = np.asarray(mtd[wksp].getNumberHistograms())

    # Have a paralle array to detectors to say what group it belongs in
    grouper   = np.full_like(spectra, -1) 


    # Loop over spectrum
    for spec_id in spectra:

        # Mean twoTheta and L2
        tt = list()
        l2 = list()

        # Loop over detectors
        for det_id in mtd[wksp].getSpectrum().getDetectorIDs():
            detector = mtd[wksp].getDetector(det_id)
            l2.append(detector.getDistance(sample))
            tt.append(detector.getTwoTheta(sample.getPos(), CentralVector) * rad2deg)

        ttAverage = np.mean(tt)
        l2Average = np.mean(l2)

        where = ttAverage / AngleStep
        grouper[spec_id] = where

    mask = grouping.utils.apply_mask(spectra, MaskIDs)
    ms, mg, groups = grouping.utils.mask_and_group(spectra, grouper, mask)
    grouping.utils.write_grouping_file(GroupingFilename, groups, instrument="NOMAD",filetype='spectra')
 

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
        print("Creating grouping file: {} ...".format(group["GroupingFilename"]["Output"]))

        # Set Q-vector flag (uses Q for the angle with CentralVector vs. detector postion)
        q_flag = group.get('UseQvectorForAngle', False)

        # Read mask IDs
        mask_ids = None
        if group["Mask"]:
            mask_ids = grouping.utils.create_id_list(**group["Mask"])

        # Get central vector
        central_vector = GenerateAxisVector(Vector=group["AxisVector"], 
                                         Rotations=group["Rotations"])    

        if "CutGroupingWithPlane" not in group:
            group["CutGroupingWithPlane"] = None

        if "AtomIdsForVisualization" not in group:
            group["AtomIdsForVisualization"] = None

        # Generate grouping filename
        if "Input" in group["GroupingFilename"]:
            GenerateGroupingFileFromSpectra(InputWorkspace=wksp,
                                            CentralVector=central_vector,
                                            AngleStep=group["AngleStep"],
                                            InitialGroupingFilename=group["GroupingFilename"]["Input"],
                                            GroupingFilename=group["GroupingFilename"]["Output"],
                                            UseQvectorForAngle=q_flag,
                                            MaskIDs=mask_ids)

        else:
            GenerateGroupingFileFromDetectors(InputWorkspace=wksp,
                                              CentralVector=central_vector,
                                              AngleStep=group["AngleStep"],
                                              GroupingFilename=group["GroupingFilename"]["Output"],
                                              UseQvectorForAngle=q_flag,
                                              MaskIDs=mask_ids,
                                              CutGroupingWithPlane=group["CutGroupingWithPlane"],
                                              AtomIdsForVisualization=group["AtomIdsForVisualization"])


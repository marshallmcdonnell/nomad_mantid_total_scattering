#!/usr/bin/env python
import json
import numpy as np
import argparse
from diagnostics import io
from diagnostics import grouping
from mantid.simpleapi import *

def GetDetectorsOfComponentName(InputWorkspace,ComponentName):
    instr_info = mtd[InputWorkspace].componentInfo()

    comp_det_list = list()
    for comp_idx in range(instr_info.size()):
        if ComponentName == instr_info.name(comp_idx):
            for det_idx in instr_info.detectorsInSubtree(comp_idx):
                comp_det_list.append(det_idx)
    return comp_det_list

def GetDetectorsOfComponentNames(InputWorkspace,ComponentNames):
    comps_det_list = list()
    for component_name in ComponentNames:
        comp_det_list = GetDetectorsOfComponentName(InputWorkspace,component_name)
        comps_det_list += comp_det_list
    return comps_det_list

def WriteAtomIdsForVisualization(wksp, grouper, pixels, filename, mask=None):
    if type(grouper) is dict:
        group_of_pixel = [ v for k, v in grouper.items() ]
    else:
        group_of_pixel = grouping.utils.revalue_array(grouper[mask])

    with open(filename, 'w') as f:
        f.write("{}\n\n".format(len(pixels)))
        for group_id, pixel_idx in zip(group_of_pixel, pixels):
            x, y, z = mtd[wksp].getDetector(int(pixel_idx)).getPos()
            print_args = {'idx' : pixel_idx,
                          'x' : x,
                          'y' : y,
                          'z' : z,
                          'group_id' : group_id
            }

            print_str="{group_id} {x} {y} {z}\n"
            f.write(print_str.format(**print_args))
    return
        

def CreateGroupingForComponentGroupingMap(InputWorkspace,GroupingInfo,InstrumentName):
    # Validate grouping map
    group_map = GroupingInfo.get('ComponentGroupingMap', None)
    filename = GroupingInfo["GroupingFilename"]["Output"]
    if group_map is None:
        print("No map for grouping file {}, skipping ...".format(filename))
        return

    print("Creating grouping file: {} ...".format(filename))

    # Get grouping
    comp_info = mtd[wksp].componentInfo()
    group_det_map = dict()
    for group_id, components_list in group_map.items():
        group_det_map[group_id] = GetDetectorsOfComponentNames(InputWorkspace,components_list)

    # Create grouper
    all_detectors = [det_id for g in group_det_map.itervalues() for det_id in g]
    grouper = dict()
    for group_id, det_ids in group_det_map.items():
        for det_id in det_ids:
            grouper[det_id] = group_id

    for group_id, detector_ids in group_det_map.items():
        for det_id in detector_ids:
            grouper[det_id] = group_id

    # Read mask
    mask_ids = None
    if "Mask" in  GroupingInfo:
        if GroupingInfo["Mask"]:
            mask_ids = grouping.utils.create_id_list(**GroupingInfo["Mask"])

    # Apply masking
    np.set_printoptions(threshold='nan')
    mask = grouping.utils.apply_mask(all_detectors, mask_ids)
    masked_data, masked_group, groups = grouping.utils.mask_and_group(all_detectors, grouper, mask)

    # Write Grouping File
    grouping.utils.write_grouping_file(filename, groups, instrument=InstrumentName)
    
    # Check if output XYZ file for visualization
    if "AtomIdsForVisualization" in GroupingInfo:
        if group["AtomIdsForVisualization"]:
            base=os.path.splitext(os.path.basename(filename))[0]
            filename = "%s.xyz" % base
            WriteAtomIdsForVisualization(InputWorkspace, grouper, masked_data, filename, mask=mask)
    return



if __name__ == "__main__":
    # Load JSON input file for groupins
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("json", type=str, help="Input JSON for grouping maps: {group1 : [components list]}")
    args = parser.parse_args()

    print("loading config from", args.json)
    with open(args.json) as handle:
        config = json.loads(handle.read())

    # Load instrument
    if "RunNumber" in config:
        wksp = "%s_%s" % (config["Instrument"], config["RunNumber"])
        Load(Filename=wksp, OutputWorkspace=wksp)

    else:
        wksp = "%s" % (config["Instrument"])
        LoadEmptyInstrument(InstrumentName=config["Instrument"], OutputWorkspace=wksp)

    # Loop over grouping files
    for group in config["Groupings"]:
        CreateGroupingForComponentGroupingMap(wksp,group, config["Instrument"])


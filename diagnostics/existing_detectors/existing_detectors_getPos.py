#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import json
import re
from mantid.simpleapi import *

import numpy as np

from diagnostics import grouping, io

#-----------------------------------------------------------------------------
# Load file

if __name__ == "__main__":
    configfile = sys.argv[1]
    print("loading config from", configfile)
    with open(configfile) as handle:
        args = json.loads(handle.read())

    # Get some output options and assign defaults if needed
    out = args["Output"]
    screen = out.get("PrintToScreen", False)
    use_atom_ids = out.get("AtomIdsForVisualization", False)
    bank_group_col = out.get("AddBankGroupingColumn", True)

    # Get pixel ids
    pixel_ids = grouping.utils.create_id_list(**args["PixelIds"])
    pixel_ids = np.asarray(list(io.utils.expand_ints(pixel_ids)))

    # Create Instrument 
    wksp='instrument'
    LoadEmptyInstrument(InstrumentName=args["Instrument"], Outputworkspace=wksp) 
    instrument = mtd[wksp].getInstrument()

    # Get xyz of pixel ids 
    xyz = np.zeros((len(pixel_ids),3))
    group_of_pixel = np.zeros(len(pixel_ids),dtype=int)
    group_pattern = re.compile(r'{instrument}/{group}(\d+)'.format(instrument=args["Instrument"],group="Group"))
    for i, pixel_idx in enumerate(pixel_ids):
        detector = instrument.getDetector(pixel_idx)
        x, y, z = detector.getPos()
        xyz[i,0] = x
        xyz[i,1] = y
        xyz[i,2] = z
        
        group_id = group_pattern.match(detector.getFullName()).group(1)
        group_of_pixel[i] = group_id

    # Output results to screen
    if screen:
        for i, (pixel_idx, (x,y,z), group_id) in enumerate(zip(pixel_ids, xyz, group_of_pixel)):
            print_args = {'i' : i, 'idx' : pixel_idx, 'x' : x, 'y' : y, 'z' : z, 'group_id' : group_id }
            if bank_group_col:
                print_str = "ID: {i} Pixel ID: {idx} XYZ: {x} {y} {z} Bank: {group_id}"
            else:
                print_str = "ID: {i} Pixel ID: {idx} XYZ: {x} {y} {z}"
            print(print_str.format(**print_args))

    # Output results to file
    atom_map = { 1 : 'S', 2 : 'D', 3 : 'H', 4 : 'O', 5 : 'N', 6 : 'C' }
    if "Filename" in out:
            filename = out["Filename"]
            with open(filename, 'w') as f:
                f.write("{}\n\n".format(len(pixel_ids)))
                for pixel_idx, (x,y,z), group_id in zip(pixel_ids, xyz, group_of_pixel):
                    print_args = {'idx' : pixel_idx, 'x' : x, 'y' : y, 'z' : z, 'group_id' : group_id }

                    if use_atom_ids:
                        print_args["idx"] = atom_map[group_id] 

                    if bank_group_col:
                        print_str="{idx} {x} {y} {z} {group_id}\n"
                    else:
                        print_str="{idx} {x} {y} {z}\n"

                    f.write(print_str.format(**print_args))

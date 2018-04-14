#!/usr/bin/env python

import argparse
import subprocess
import collections
import xml.etree.ElementTree

from diagnostics import io
from diagnostics import grouping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('filename', type=str,
                        help="Grouping filename (XML) to parse for visualization")
    parser.add_argument('-o', '--output', type=str, default="out.xyz",
                        help="Ouput filename (XYZ) for VMD visualization")
    parser.add_argument('--group-ids', nargs='*', type=int, dest="group_ids",
                        help="List of group IDs to visualize")
    parser.add_argument('--vmd', action="store_true", default=False,
                        help="Launch VMD directly")
    parser.add_argument('--elements', nargs='*', type=str,
                        help="List of elements to use for group ids in XYZ file.")
    args = parser.parse_args()

    grouping = xml.etree.ElementTree.parse(args.filename)

    # Get all group ids in file
    all_group_ids = list()
    for child in grouping.getroot():
        all_group_ids.append(int(child.attrib['ID']))

    # Get group ids to visualize
    group_ids = args.group_ids
    if not group_ids:
        group_ids = all_group_ids

    for gid in group_ids:
        if gid not in all_group_ids:
            print("\nWARNING: Did not find Group ID %d" % gid)
            print("         This will break --elements argument\n")

    # Get pixels for group ids
    num_pixel_ids = 0
    groups_map = collections.OrderedDict()
    for child in grouping.getroot():
        gid = int(child.attrib['ID'])
        if gid in group_ids:
            detids_compressed = child.find('detids').text
            pixel_ids = list(io.utils.expand_ints(detids_compressed))
            groups_map[gid] = pixel_ids
            num_pixel_ids += len(pixel_ids)

    # Use elements defined vs. just group IDs
    group_id_map = collections.OrderedDict()
    for k in  groups_map.keys():
        group_id_map[k] = k

    if args.elements:
        assert(len(group_id_map.keys()) == len(args.elements))
        for gid, element in zip(group_ids, args.elements):
            group_id_map[gid] = element
        
    # Get instrument and detectors
    from mantid.simpleapi import LoadEmptyInstrument, mtd
    wksp = "instrument"
    LoadEmptyInstrument(InstrumentName="NOMAD", OutputWorkspace=wksp)
    instrument = mtd[wksp].getInstrument()

    # Output XYZ file
    with open(args.output, 'w') as f:
        f.write("{}\n\n".format(num_pixel_ids))
        for group_id, pixel_ids in groups_map.items():
            for pixel_idx in pixel_ids:
                x, y, z = instrument.getDetector(pixel_idx).getPos()
                print_args = {'idx' : pixel_idx,
                              'x' : x,
                              'y' : y,
                              'z' : z,
                              'group_id' : group_id_map[group_id]
                }

                print_str="{group_id} {x} {y} {z}\n"
                f.write(print_str.format(**print_args))

    if args.vmd:
        subprocess.call(["vmd", args.output])

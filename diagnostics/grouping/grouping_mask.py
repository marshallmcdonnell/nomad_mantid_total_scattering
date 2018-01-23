#!/usr/bin/env python

import numpy as np
import argparse
from collections import Counter

from diagnostics import io

#-----------------------------------------------------
# Create mask from string of indices i.e. "0,1,2-10"

def create_mask(array,mask_ids):
    mask= np.ones(len(array), dtype=bool)
    if not mask_ids:
        return mask
    ind2remove = set(list(io.utils.expand_ints(mask_ids)))
    ind2remove = [ i for i in ind2remove if i < len(mask)]
    mask[ind2remove] = False
    return mask

#-----------------------------------------------------
# Revalue an array to take out gaps in increment

def revalue_grouping(array):
    revalued = np.zeros_like(array)
    idx_start = 0
    counter = 0
    for key, number_values in sorted(Counter(array).items()):
        idx_stop = idx_start + number_values
        revalued[idx_start:idx_stop] = counter
        counter += 1
        idx_start = idx_stop
    return revalued

#-----------------------------------------------------
# Mask data and re-group (w/o gaps) applying mask

def mask_and_group(data, grouping, mask):
    # Relabel grouping so groups have no gaps in Group IDs
    masked_grouping = revalue_grouping(grouping[mask])

    # Get masked data as array
    masked_data = data[mask]

    # Get Groups
    groups = list()
    for group_num in np.unique(masked_grouping):
        groups.append( masked_data[masked_grouping == group_num] )

    return masked_data, masked_grouping, groups

if __name__ == "__main__":
    #-----------------------------------------------------
    # Parse input

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-ids", type=str, default=None, dest="mask_ids",
                        help="List of pixels to mask")
    parser.add_argument("--mask-ids-file", type=str, default=None,dest="mask_ids_file",
                        help="Filename with list of pixels to mask")
    args = parser.parse_args() 

    #-----------------------------------------------------
    # Create pixel layout

    pixels_per_tube=128
    tubes_per_8pack=8
    num_8packs=99
    tot_pixels=num_8packs * tubes_per_8pack * pixels_per_tube

    #-----------------------------------------------------
    # Create the unique pixel ids (i=0 -> N, N=total pixels) in data
    # and the grouping of the pixels 
    data = np.arange(0,tot_pixels,dtype=int)
    grouping = np.repeat( np.arange(pixels_per_tube*tubes_per_8pack), num_8packs)

    # Create the mask to apply to both data and grouping
    mask_ids=str()
    if args.mask_ids:
        mask_ids += args.mask_ids
    if args.mask_ids_file:
        with open(args.mask_ids_file, 'r') as f:
            mask_ids_from_file = ",".join(line.strip() for line in f)
            if mask_ids:
                mask_ids += ','+mask_ids_from_file
            else:
                mask_ids = mask_ids_from_file

    mask = create_mask(data,mask_ids)

    # Print Results
    io.utils.print_array("Data", data)
    io.utils.print_array("Grouping", grouping)
    io.utils.print_array("Mask", mask)

    masked_data, masked_grouping, groups = mask_and_group(data, grouping, mask)
    io.utils.print_array("Masked Data", data[mask])
    io.utils.print_array("Masked Group", grouping[mask])
    io.utils.print_array("New Masked Group", masked_grouping)

    for i, group in enumerate(groups):
        print("Group #: {} Data: {}".format(i, group))


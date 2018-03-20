#!/usr/bin/env python

import numpy as np
import argparse
from collections import Counter

from diagnostics import io

#-----------------------------------------------------
# Apply a mask to array from string of indices i.e. "0,1,2-10"

def apply_mask(array, mask_ids):
    mask = np.ones(len(array), dtype=bool)
    if not mask_ids:
        return mask
    ind2remove = set(list(io.utils.expand_ints(mask_ids)))
    ind2remove = [i for i in ind2remove if i < len(mask)]
    mask[ind2remove] = False
    return mask

#-----------------------------------------------------
# Create id list from string and/or file with list of ids

def create_id_list(List=None, Filename=None):
    all_ids = str()
    if List:
        all_ids += List

    if Filename:
        with open(Filename, 'r') as f:
            ids_from_file = ",".join(line.strip() for line in f)
            if all_ids:
                all_ids += ',' + ids_from_file
            else:
                all_ids = ids_from_file

    return all_ids

#-----------------------------------------------------
# Revalue an array to take out gaps in increment


def revalue_array(array):
    old_group_nums = Counter(array).keys()
    new_group_nums = range(len(old_group_nums))
    revalue_map = {
        old: new for (
            old,
            new) in zip(
            old_group_nums,
            new_group_nums)}
    revalued = np.copy(array)
    for k, v in revalue_map.items():
        revalued[array == k] = v
    return revalued

#-----------------------------------------------------
# Mask data and re-group (w/o gaps) applying mask


def mask_and_group(data, grouper, mask):
    # Relabel grouper so groups have no gaps in Group IDs
    masked_grouper = revalue_array(grouper[mask])

    # Get masked data as array
    masked_data = data[mask]

    # Get Groups
    groups = list()
    for group_num in np.unique(masked_grouper):
        groups.append(masked_data[masked_grouper == group_num])

    return masked_data, masked_grouper, groups


def write_grouping_file(filename,groups,instrument="NOMAD"):
    handle = file(filename, 'w')
    handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
    handle.write('<detector-grouping instrument="%s">\n' % instrument)

    for groupnum, group in enumerate(groups):
        handle.write('<group ID="%d">\n' % (groupnum + 1))
        handle.write('<detids>%s</detids>\n' % (io.utils.compress_ints(group)))
        handle.write('</group>\n')
    handle.write('</detector-grouping>\n')

if __name__ == "__main__":
    #-----------------------------------------------------
    # Parse input

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-ids", type=str, default=None, dest="mask_ids",
                        help="List of pixels to mask")
    parser.add_argument(
        "--mask-ids-file",
        type=str,
        default=None,
        dest="mask_ids_file",
        help="Filename with list of pixels to mask")
    args = parser.parse_args()

    #-----------------------------------------------------
    # Create pixel layout

    pixels_per_tube = 128
    tubes_per_8pack = 8
    num_8packs = 99
    tot_pixels = num_8packs * tubes_per_8pack * pixels_per_tube

    #-----------------------------------------------------
    # Create the unique pixel ids (i=0 -> N, N=total pixels) in data
    # and the grouper of the pixels
    data = np.arange(0, tot_pixels, dtype=int)
    grouper = np.repeat(
        np.arange(
            pixels_per_tube *
            tubes_per_8pack),
        num_8packs)

    # Create the mask to apply to both data and grouper
    mask_ids = str()
    if args.mask_ids:
        mask_ids += args.mask_ids
    if args.mask_ids_file:
        with open(args.mask_ids_file, 'r') as f:
            mask_ids_from_file = ",".join(line.strip() for line in f)
            if mask_ids:
                mask_ids += ',' + mask_ids_from_file
            else:
                mask_ids = mask_ids_from_file

    mask = apply_mask(data, mask_ids)

    # Print Results
    io.utils.print_array("Data", data)
    io.utils.print_array("Grouping", grouper)
    io.utils.print_array("Mask", mask)

    masked_data, masked_grouper, groups = mask_and_group(data, grouper, mask)
    io.utils.print_array("Masked Data", data[mask])
    io.utils.print_array("Masked Group", grouper[mask])
    io.utils.print_array("New Masked Group", masked_grouper)

    for i, group in enumerate(groups):
        print("Group #: {} Data: {}".format(i, group))

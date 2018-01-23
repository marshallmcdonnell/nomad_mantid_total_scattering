#!/usr/bin/env python

import numpy as np
from collections import Counter
from itertools import chain
import argparse

#-----------------------------------------------------
# Parse input

parser = argparse.ArgumentParser()
parser.add_argument("--mask-ids", type=str, default=None, dest="mask_ids",
                    help="List of pixels to mask")
parser.add_argument("--mask-ids-file", type=str, default=None,dest="mask_ids_file",
                    help="Filename with list of pixels to mask")
args = parser.parse_args() 

#-----------------------------------------------------
# Printing function

def print_array(title, array, num_dashes=35):
    print("-"*num_dashes)
    print("{}: array of length = {}".format(title, len(array)))
    print(array)
    print("-"*num_dashes)

#-----------------------------------------------------
# Revalue an array to take out gaps in increment

def revalueGrouping(array):
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
# Function to expand string of ints with dashes
# Ex. "1-3, 8-9, 12" -> [1,2,3,8,9,12] 

def expandInts(s):
    spans = (el.partition('-')[::2] for el in s.split(','))
    ranges = (xrange(int(s), int(e) + 1 if e else int(s) + 1) for s, e in spans)
    all_nums = chain.from_iterable(ranges)
    return all_nums

#-----------------------------------------------------------------------------------
# Function to compress list of ints with dashes
# Ex. [1,2,3,8,9,12] -> 1-3, 8-9, 12

def compressInts(line_nums):
    seq = []
    final = []
    last = 0

    for index, val in enumerate(line_nums):

        if last + 1 == val or index == 0:
            seq.append(val)
            last = val
        else:
            if len(seq) > 1:
               final.append(str(seq[0]) + '-' + str(seq[len(seq)-1]))
            else:
               final.append(str(seq[0]))
            seq = []
            seq.append(val)
            last = val

        if index == len(line_nums) - 1:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq)-1]))
            else:
                final.append(str(seq[0]))

    final_str = ', '.join(map(str, final))
    return final_str


#-----------------------------------------------------
# Create pixel layout

pixels_per_tube=10
tubes_per_8pack=2
num_8packs=3
tot_pixels=num_8packs * tubes_per_8pack * pixels_per_tube

#-----------------------------------------------------
# Create the unique pixel ids (i=0 -> N, N=total pixels) in data
# and the grouping of the pixels 
data = np.arange(0,tot_pixels,dtype=int)
grouping = np.repeat( np.arange(pixels_per_tube*tubes_per_8pack), num_8packs)

# Create the mask to apply to both data and grouping
mask= np.ones(len(data), dtype=bool)

if args.mask_ids:
    ind2remove = list(expandInts(args.mask_ids))
if args.mask_ids_file:
    with open(args.mask_ids_file, 'r') as f:
        mask_ids = ",".join(line.strip() for line in f)
        ind2remove += expandInts(mask_ids)
mask[list(set(ind2remove))] = False

# Print Results
print_array("Data", data)
print_array("Grouping", grouping)
print_array("Mask", mask)
print_array("Masked Data", data[mask])
print_array("Masked Group", grouping[mask])

# Relabel grouping so groups have no gaps in Group IDs
new_grouping = revalueGrouping(grouping[mask])
print_array("New Masked Group", new_grouping)

# Get masked data as array
masked_data = data[mask]

# Get Groups
groups = list()
for group_num in np.unique(new_grouping):
    print("Group #: {} Data: {}".format(group_num, masked_data[ new_grouping == group_num]))

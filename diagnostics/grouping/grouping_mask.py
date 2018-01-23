#!/usr/bin/env python

import numpy as np
from collections import Counter
import itertools
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
mask= np.ones(len(data), dtype=bool)
ind2remove=[0, 3,4,5]
mask[ind2remove] = False

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

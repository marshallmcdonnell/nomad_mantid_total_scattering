#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import os
import json
import numpy as np
import argparse
from mantid.simpleapi import *

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("-p", "--print_separate_lines", action="store_true", default=False)
args = parser.parse_args()


#-----------------------------------------------------------------------------------
# Function to compress list of ints with dashes
# Ex. [1,2,3,8,9,12] -> 1-3, 8-9, 12

def get_line_numbers_concat(line_nums):
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


#-----------------------------------------------------------------------------------
# Print with each entry on individual lines
def print_separate_lines(title, ids, num_dashes=35):
    print ("-"*num_dashes)
    print (title)
    print ("-"*num_dashes)
    for group in get_line_numbers_concat(ids).split(','):
        print(group)

def print_standard(title, ids, num_dashes=35):
    print ("-"*num_dashes)
    print ("{}: {}".format( title, get_line_numbers_concat(ids)))
    print ("-"*num_dashes)

#-----------------------------------------------------------------------------------
# Load file



# Load aligned workspace
wksp = Load(Filename=args.filename)



non_existent_ids = list()
existent_ids = list()
for i in range(wksp.getNumberHistograms()):
    y = wksp.readY(i)
    x = wksp.readX(i)
    e = wksp.readE(i)

    if np.count_nonzero(y,axis=0) == 0:
        non_existent_ids.append(i)
    else:
        existent_ids.append(i)

if args.print_separate_lines:
    title="Non-Existing IDs"
    print_separate_lines(title, non_existent_ids)
    title="Existing IDs"
    print_separate_lines(title, existent_ids)

else:
    title="Non-Existing IDs"
    print_standard(title, non_existent_ids)
    print()
    title="Existing IDs"
    print_standard(title, existent_ids)



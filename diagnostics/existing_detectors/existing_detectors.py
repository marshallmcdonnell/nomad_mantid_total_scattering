#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import sys
import os
import json
import numpy as np
from mantid.simpleapi import *

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
# Load file

configfile = sys.argv[1]
print("loading config from", configfile)
with open(configfile) as handle:
    config = json.loads(handle.read())

filename = config['Filename']
save_dir = config.get('SaveDirectory', '/tmp')

# Load aligned workspace
wksp = Load(Filename=filename)



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

num_dashes=35
print ("-"*num_dashes)
print ("Non-Existing IDs: {}".format(get_line_numbers_concat(non_existent_ids)))
print ("-"*num_dashes)
print ("Existing IDs:     {}".format(get_line_numbers_concat(existent_ids)))
print ("-"*num_dashes)


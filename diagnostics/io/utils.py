import numpy as np
import itertools

#-----------------------------------------------------
# Printing function


def print_array(title, array, num_dashes=35):
    print("-" * num_dashes)
    print("{}: array of length = {}".format(title, len(array)))
    print(array)
    print("-" * num_dashes)

#-----------------------------------------------------
# Function to expand string of ints with dashes
# Ex. "1-3, 8-9, 12" -> [1,2,3,8,9,12]


def expand_ints(s):
    spans = (el.partition('-')[::2] for el in s.split(','))
    ranges = (xrange(int(s), int(e) + 1 if e else int(s) + 1)
              for s, e in spans)
    all_nums = itertools.chain.from_iterable(ranges)
    return all_nums

#-------------------------------------------------------------------------
# Function to compress list of ints with dashes
# Ex. [1,2,3,8,9,12] -> 1-3, 8-9, 12


def compress_ints(line_nums):
    seq = []
    final = []
    last = 0

    for index, val in enumerate(line_nums):

        if last + 1 == val or index == 0:
            seq.append(val)
            last = val
        else:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq) - 1]))
            else:
                final.append(str(seq[0]))
            seq = []
            seq.append(val)
            last = val

        if index == len(line_nums) - 1:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq) - 1]))
            else:
                final.append(str(seq[0]))

    final_str = ', '.join(map(str, final))
    return final_str

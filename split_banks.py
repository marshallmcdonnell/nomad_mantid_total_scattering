#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


def split_banks(f):
    lines = f.readlines()

    is_first = True
    nrows = 0
    nsets = 0
    for l in lines:
        ncols = len(l.strip().split())
        if ncols == 1:
            nrows = 0
            nsets += 1
        else:
            nrows += 1

    datasets = np.empty([nsets, nrows, 3])

    for i in range(nsets):
        for j in range(nrows):
            line = lines[i * (nrows + 1) + j + 1].strip()
            x, y, err = line.split()
            datasets[i, j, :] = [x, y, err]

    return datasets


parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+', help="Filename with bank info")
parser.add_argument(
    '--stem_name',
    default=None,
    help="Stem name for output files if different from original filename.")
parser.add_argument('--error', action="store_true", default=False,
                    help="Add 3rd column to output with error.")
args = parser.parse_args()
print(args)

datasets = list()
for i, filename in enumerate(args.filenames):
    with open(filename, 'r') as f:
        dataset = split_banks(f)
        datasets.append(dataset)

dmins = np.asarray([.61, .30, .13, .13, .13, 1.07])
dmaxs = np.asarray([10.31, 5.37, 2.67, 1.67, 1.57, 17.95])

qmins = 2. * np.pi / dmaxs
qmaxs = 2. * np.pi / dmins

# Loop datasets for each filename
for dataset, filename in zip(datasets, args.filenames):

    fname, ext = (os.path.splitext(os.path.basename(filename)))

    # Loop over the bank-by-bank data
    for i, (bank, qmin, qmax) in enumerate(zip(dataset, qmins, qmaxs)):

        # create a bank-by-bank filename for output
        if args.stem_name:
            bank_filename = "%s_%d.dat" % (args.stem_name, i)
        else:
            bank_filename = "%s_%d.dat" % (fname, i)

        with open(bank_filename, 'w') as bf:

            # make output for the x, y that are within the qmin, qmax range
            output = list()
            for x, y, yerr in bank:
                if x >= qmin and x <= qmax:
                    if args.error:
                        output.append(x, y, yerr)
                    else:
                        output.append((x, y))
            bf.write("%d \n" % len(output))
            bf.write("From filename %s bank: %d\n" % (filename, i))
            for x, y in output:
                bf.write("%f.4 %f.4 \n" % (x, y))

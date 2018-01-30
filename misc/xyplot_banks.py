#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_banks(f, title=None, banks_list=None, line_style=None, error=None):
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
            if error:
                datasets[i, j, :] = [x, y, err]
            else:
                datasets[i, j, :] = [x, y, len(x) * 0.]

    if banks_list is None:
        banks_list = range(nsets)

    banks_list = [int(i) for i in banks_list]

    color_line_styles = {0: 'k',
                         1: 'r',
                         2: 'b',
                         3: 'g',
                         4: 'y',
                         5: 'c'}

    for i in range(nsets):
        if i in banks_list:
            myLabel = 'bank: ' + str(i) + ' ' + title.split('.')[0]
            if error:
                plt.errorbar(datasets[i, :, 0], datasets[i, :, 1],
                             yerr=datasets[i, :, 2],
                             fmt=line_style + color_line_styles[i],
                             markersize=4,
                             label=myLabel)

            else:
                plt.plot(datasets[i, :, 0], datasets[i, :, 1],
                         line_style + color_line_styles[i],
                         markersize=4,
                         label=myLabel)

    return datasets


parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+', help="Filename with bank info")
parser.add_argument('-b', '--banks', nargs='+', help='Banks to plot')
parser.add_argument(
    '--error',
    action="store_true",
    default=False,
    help='Plot error bars')
args = parser.parse_args()
print(args)

line_styles = {0: '-',
               1: '--',
               2: ':',
               3: '-x',
               4: '--x',
               5: ':x'}

datasets = list()
for i, filename in enumerate(args.filenames):
    with open(filename, 'r') as f:
        if len(args.filenames) == 1:
            plot_banks(
                f,
                title=filename,
                banks_list=args.banks,
                line_style=line_styles[i],
                error=args.error)
        else:
            dataset = plot_banks(
                f,
                title=filename,
                banks_list=args.banks,
                line_style=line_styles[i],
                error=args.error)
            datasets.append(dataset)

plt.legend(loc='best')
plt.xlabel("Q (angstroms^-1")
# plt.xlabel("Wavelength")
plt.ylabel("I(Q) (arb. units)")
#lt.title("Carpenter Corrections (Abs. + Mult. Scat.)")
plt.show()

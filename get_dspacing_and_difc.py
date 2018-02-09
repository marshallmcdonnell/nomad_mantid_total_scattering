#!/usr/bin/env python

import argparse
import numpy as np
from mantid.simpleapi import *

from diagnostics import io

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("calibration", type=str)
parser.add_argument("-s", "--spectrum-list", type=str, dest="spectrum_list")
parser.add_argument("-p", "--peak-parameters", nargs="+", action="append", 
                    default=list(), dest="peak_parameters", type=float,
                    help="Xmin, CurrentPeakPosition, XMax, DesiredPeakPosition")
args = parser.parse_args()

spectrum_list = io.utils.expand_ints(args.spectrum_list)


peaks = list()
windows = list()
for i, p in enumerate(args.peak_parameters):
    peaks.append(p[1])
    windows += [p[0], p[2]]

wksp = Load(Filename=args.filename)
print "Number of histograms in workspace:", wksp.getNumberHistograms()

LoadDiffCal(InputWorkspace=wksp,
            Filename=args.calibration,
            WorkspaceName='NOM_cal',
            TofMin=300, TofMax=16666)

original_data = dict()
for spectrum in spectrum_list:
    table = FindPeaks(InputWorkspace=wksp,
                      WorkspaceIndex=spectrum,
                      BackgroundType='Linear',
                      HighBackground=True,
                      PeakPositions=peaks,
                      FitWindows=windows
    )

    spectrum_dict = dict()
    for row_num in range(table.rowCount()):
        row = table.row(row_num)
        spectrum_dict[row_num] = { 'centre' : table.row(row_num)['centre'],
                                   'width'  : table.row(row_num)['width'],
                                   'height' : table.row(row_num)['height']
        }

    original_data[spectrum] = spectrum_dict

# Plotting
import matplotlib.pyplot as plt

f, axarr = plt.subplots(2, sharex=True)

def add_plot(axis, data, key, peak_id):
    x = original_data.keys()
    y = [ values[peak_id][key] for spectrum, values in orginal_data.items() ]
    axis.plot(x,y)
    return axis

for peak_id in range(len(peaks)):
    axarr[0] = add_plot(axarr[0], original_data, 'centre', peak_id)

for peak_id in range(len(peaks)):
    axarr[1] = add_plot(axarr[1], original_data, 'height', peak_id)
plt.show()

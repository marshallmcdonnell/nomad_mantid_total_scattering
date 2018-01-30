#!/usr/bin/env python

import argparse
from mantid.simpleapi import *

from diagnostics import io

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--spectrum-list", type=str, dest="spectrum_list")
args = parser.parse_args()

spectrum_list = io.utils.expand_ints(args.spectrum_list)

wksp = Load(Filename=args.filename)

print wksp.getNumberHistograms()
for spectrum in spectrum_list:
    table = FindPeaks(InputWorkspace=wksp,
                      WorkspaceIndex=spectrum,
                      HighBackground=False,
                      BackgroundType="Quadratic",
                      MinimumPeakHeight=50)

    print "Found %d peaks" % table.rowCount()
    for row_num in range(table.rowCount()):

        row = table.row(row_num)
        print("Peak %d {Centre: %.3f, width: %.3f, height: %.3f }" %
              (row_num, row["centre"], row["width"], row["height"]))

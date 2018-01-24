#!/usr/bin/env python

import argparse
from mantid.simpleapi import *

from diagnostics import io

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--spec-list", type=str, dest="spectrum_list")
args = parser.parse_args()

spectrum_list = io.utils.expand_ints(args.spectrum_list)

wksp = Load(Filename=filename)

for spectrum in spectrum_list:
    table = FindPeaks(InputWorkspace='ws',
                      WorkspaceIndex=spectrum)

    row = table.row(0)

    print("Peak 1 {Centre: %.3f, width: %.3f, height: %.3f }" %
          (row["centre"], row["width"], row["height"]))

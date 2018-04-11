#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from h5py import File

def extract(hdf_file,path):
    data = File(hdf_file, 'r')
    return data[path]

def extract_xy(hdf_file, xpath, ypath):
    x = extract(hdf_file, xpath)
    y = extract(hdf_file, ypath)
    return x, y

def save_xy(filename, xdata, ydata):
    assert(len(xdata) == len(ydata))
    with open(filename, 'w') as f:
        f.write("%d\n\n" % len(xdata))
        for x, y in zip(xdata, ydata):
            f.write("%f %f\n" % (x, y))

#-------------------------------------------------------------------------
# MAIN - NOM_pdf
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract info from HDF5 file")
    parser.add_argument('-i', '--input', help='HDF5 file to read from')
    parser.add_argument('-x', '--xpath', help='Path in HDF5 file to X data')
    parser.add_argument('-y', '--ypath', help='Path in HDF5 file to Y data')
    parser.add_argument('-o', '--output', help='File to save XY data')
    args = parser.parse_args()

    if not args.input or not args.xpath or not args.ypath:
        raise Exception("Must specificy input file, xpath and ypath")

    x, y = extract_xy(args.input, args.xpath, args.ypath)
    save_xy(args.output, x, y)

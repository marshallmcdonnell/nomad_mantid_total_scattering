#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from h5py import File
import numpy as np

def extract(hdf_file,path, index=None):
    data = File(hdf_file, 'r')
    base=path.split('/')[1]
    print(data[base+"/title"].value)
    if index is not None:
        return data[path][index]
    return data[path].value

def extract_xy(hdf_file, xpath, ypath, **kwargs):
    x = extract(hdf_file, xpath)
    y = extract(hdf_file, ypath, **kwargs)
    y = np.insert(y,0,y[0]) # hack for histogram xaxis
    return x, y

def save_xy(filename, xdata, ydata):
    #assert(len(xdata) == len(ydata))
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
    parser.add_argument('--yindex', type=int, default=None, 
                        help='If ypath is 2D, extract just the yindex row')
    parser.add_argument('-o', '--output', help='File to save XY data')
    args = parser.parse_args()

    if not args.input or not args.xpath or not args.ypath:
        raise Exception("Must specificy input file, xpath and ypath")

    x, y = extract_xy(args.input, args.xpath, args.ypath, index=args.yindex)
    save_xy(args.output, x, y)

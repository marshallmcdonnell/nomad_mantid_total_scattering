#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import sys
import os
from h5py import File
import numpy as np

def extract(hdf_file,path, index=None):
    data = File(hdf_file, 'r')
    base=path.split('/')[1]
    #print(data[base+"/title"].value)
    if index is not None:
        return data[path][index]
    return data[path].value

def extract_path_from_title(hdf_file, title, title_path="title", wksp_path="workspace"):
    data = File(hdf_file, 'r')
    choices = list()
    for name, group in data.iteritems():
        index = os.path.join(name, title_path)
        if data[index].value == title:
            xpath = os.path.join("/",name, wksp_path, "axis1")
            ypath = os.path.join("/", name, wksp_path, "values")
            return xpath, ypath
        else:
            choices.append(data[index].value)

    print("ERROR: Did not find a workspace with title: %s", title)
    print("       These are the workspaces titles found in this file")
    for c in sorted(choices):
        print("      %s" % c)
    sys.exit("Stopping...") 
    
 
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
    parser.add_argument('--title', type=str, default=None, 
                        help='From ypath, use this title to extract')
    parser.add_argument('-o', '--output', help='File to save XY data')
    args = parser.parse_args()

    if not args.input:
        raise Exception("Must specificy input file...")

    if (not args.xpath or not args.ypath) and not args.title:
        raise Exception("Must specificy xpath and ypath or workspace title")
    
    if (args.xpath or args.ypath) and args.title:
        print("WARNING: Specified  both path and title (only use one). Proceeding with title...")

    if args.title:
        args.xpath, args.ypath = extract_path_from_title(args.input,args.title)

    x, y = extract_xy(args.input, args.xpath, args.ypath, index=args.yindex)
    save_xy(args.output, x, y)

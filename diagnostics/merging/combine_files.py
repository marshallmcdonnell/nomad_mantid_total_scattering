#!/usr/bin/env python                                                                       
from __future__ import (absolute_import, division, print_function)                          
                                                    
import inspect        
import argparse                  
import numpy as np                              
import pandas as pd


# -------------------------------------#
# Utilities

def get_data(filename, skiprows=0, skipfooter=0, xcol=0, ycol=1):
    # Setup domain 
    return pd.read_csv(filename,sep=r"\s*",
                       skiprows=skiprows,
                       skipfooter=skipfooter,
                       usecols=[xcol,ycol],
                       names=['x','y'],
                       engine='python')

def add_dataset(df_main, df_new):
    y = df_new.values[:,1]                              
    x = df_new.values[:,0]
    df = pd.DataFrame(y, index=x)
    return pd.concat([df_main, df], axis=1)

# -------------------------------------#
# Main function CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames', nargs='*',default=list(),
                        help='Multiple filenames. The x and y col are set by --xcol and --ycol. Default x=0, y=1')
    parser.add_argument('-s', '--skiprows', type=int, default=0,
                        help='Number of rows to skip in datasets')
    parser.add_argument('-t', '--trim', type=int, default=0,
                        help='Number of rows to trim off end in datasets')
    parser.add_argument('-f', '--filename', nargs='*', action="append",default=list(),
                        help='Filename, x-col, y-col')
    parser.add_argument('-x', '--xcol', type=int, default=0,
                        help='Set x-col for multiple filenames (--filenames <filenames>)')
    parser.add_argument('-y', '--ycol', type=int, default=1,
                        help='Set y-col for multiple filenames (--filenames <filenames>)')
    parser.add_argument('-o', '--output', type=str, default='out.csv',
                        'Ouptput csv file')

    args = parser.parse_args()

    kwargs = {"skiprows" : args.skiprows,
              "trim" : args.trim}

    df_total = pd.DataFrame([])
    for f in args.filenames:
        df = get_data(f,xcol=args.xcol,ycol=args.ycol,**kwargs)
        df_total = add_dataset(df_total, df)

    for f in args.filename:
        xcol = int(f[1])
        ycol = int(f[2])
        if len(f) > 3:
            kwargs['sep'] = f[3]
            if kwargs['sep'] == 'csv':
                kwargs['sep'] = r"\s*,\s*"
            else:
                kwargs['sep'] = r"\s*"

        df = get_data(f,xcol=args.xcol,ycol=args.ycol,**kwargs)
        df_total = add_dataset(df_total, df)

    df_total_fillna = df_total.fillna(0) 
    df_total_fillna.to_csv(args.output)


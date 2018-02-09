#!/usr/bin/env python

import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *

def add_plot(axis, data, key, peak_id,peak, color=''):
    x = data.keys()
    y = [ values[peak_id][key] for spectrum, values in data.items() ]
    axis.plot(x,y, color+'-', label='dspacing: %.2f' % peak)
    return axis
    
def plot_figures(list_of_plots, data, peaks, difc_data=None,difa_data=None,tzero_data=None,mask_data=None,target_peaks=None,difc_key_order=None):
    x = data.keys()
    nplots = len(list_of_plots)
    f, axarr = plt.subplots(nplots, sharex=True)
    i = 0
    
    colors = ['r', 'g', 'b', 'k']
    if 'centre' in list_of_plots:
        for peak_id, (peak, color) in enumerate(zip(peaks, colors)):
            axarr[i] = add_plot(axarr[i], data, 'centre', peak_id, peak, color=color)
            if target_peaks is not None:
                axarr[i].plot(x, len(x)*[target_peaks[peak_id]], color+'--', label='target for: %.2f' % peak)
            axarr[i].set(ylabel='d-spacing')
            axarr[i].legend()
        i += 1
             
    if 'height' in list_of_plots:
        for peak_id, peak in enumerate(peaks):
            axarr[i] = add_plot(axarr[i], data, 'height', peak_id, peak)
            axarr[i].set(ylabel='I(d) arb. units')
            axarr[i].legend()
        i += 1
             

    if 'difc' in list_of_plots and difc_data:
         keys = difc_data.keys()
         if difc_key_order:
             keys = difc_key_order
         for key in keys:
            axarr[i].plot(x, difc_data[key], '-', label=key)
            axarr[i].set(ylabel='difc')
            axarr[i].legend()
         i += 1
            
    for itype, idata in zip(['difa', 'tzero', 'mask'], [difa_data, tzero_data, mask_data]):
        if itype in list_of_plots and idata is not None:
            axarr[i].plot(x,idata, '-')
            axarr[i].set(ylabel=itype)
            i += 1
            
    axarr[i-1].set(xlabel='pixel ID')
    plt.show()

def expand_ints(s):
    spans = (el.partition('-')[::2] for el in s.split(','))
    ranges = (xrange(int(s), int(e) + 1 if e else int(s) + 1)
              for s, e in spans)
    all_nums = itertools.chain.from_iterable(ranges)
    return all_nums

#--------------------------------------------------------------------------------------------------------------#
# Inputs

filename = "/SNS/NOM/shared/diagnostics/NOM_94730_aligned_focussed_existing_pixels.nxs"
calibration_filename = "/SNS/NOM/shared/CALIBRATION/2017_2_1B_CAL/NOM_d94806_2017_07_24_shifter_open2_12vx6h.h5"
component="bank37"

spectrum_list = "14344-14598,14600-14726,14728-14854,14856-14982,14984-15110,15112-15238,15240-15366"
original_peaks = [0.9,1.1,1.80]
original_windows = [0.8,0.96,0.96,1.2,1.6,2.1]

target_spectrum=17600
target_peaks = [1.1,1.25,2.1]
target_windows = [1,1.2,1.2,1.4,1.8,2.4]

# Load files
nom_wksp = "NOM_wksp"
Load(Filename=filename, OutputWorkspace=nom_wksp)

name="NOM"
LoadDiffCal(Filename=calibration_filename,
                   WorkspaceName=name,
                   MakeGroupingWorkspace=False,
                   InputWorkspace=nom_wksp,
                   TofMin=300, TofMax=16666
)
cal_wksp=name+"_cal"
mask_wksp=name+"_mask"
                   
# Get spectrum and convert to 0-index from 1-index
spectrum_list = expand_ints(spectrum_list)
spectrum_list = [x-1 for x in spectrum_list]

# Get peaks for bad spectrum
original_data = dict()
for spectrum in spectrum_list:
    table = FindPeaks(InputWorkspace=nom_wksp,
                      WorkspaceIndex=spectrum,
                      BackgroundType='Linear',
                      HighBackground=True,
                      PeakPositions=original_peaks,
                      FitWindows=original_windows
    )

    spectrum_dict = dict()
    for row_num in range(table.rowCount()):
        row = table.row(row_num)
        spectrum_dict[row_num] = { 'centre' : table.row(row_num)['centre'],
                                   'width'  : table.row(row_num)['width'],
                                   'height' : table.row(row_num)['height']
        }

    detID = mtd[nom_wksp].getDetector(spectrum).getID()
    original_data[detID] = spectrum_dict
    

# Get DIFC, DIFA, and TZERO
difc_list = mtd[cal_wksp].column('difc')
difa_list = mtd[cal_wksp].column('difa')
tzero_list = mtd[cal_wksp].column('tzero')
mask_list = mtd[mask_wksp].extractY()

detIDs= original_data.keys()
original_difc = np.array( [ difc_list[x] for x in detIDs])
original_difa = np.array( [ difa_list[x] for x in detIDs ])
original_tzero = np.array( [ tzero_list[x] for x in detIDs ])
original_mask = np.array( [mask_list[x] for x in detIDs])
difc_dict = { 'original' : original_difc }

# Get Target d-spacing
table = FindPeaks(InputWorkspace=nom_wksp,
                             WorkspaceIndex=target_spectrum,
                             BackgroundType='Linear',
                             HighBackground=True,
                             PeakPositions=target_peaks,
                             FitWindows=target_windows
)

target_spectrum_dict = dict()
for row_num in range(table.rowCount()):
        row = table.row(row_num)
        target_spectrum_dict[row_num] = { 'centre' : table.row(row_num)['centre'],
                                                                 'width'  : table.row(row_num)['width'],
                                                                 'height' : table.row(row_num)['height']
}

target_dspacing= [ target_spectrum_dict[row]['centre'] for row in range(table.rowCount()) ]

# Get New DIFC
dspacing = list()
for key, spectrum_dict in original_data.items():
    dspacing.append([ peak_info['centre'] for peak_id, peak_info in spectrum_dict.items() ])
original_dspacing = np.array(dspacing)

new_difc = (original_dspacing.T * original_difc).T / target_dspacing
difc_key_order = list()
for peak_id, peak in enumerate(original_peaks):
    key = 'dspacing: %.2f' % peak
    difc_key_order.append(key)
    difc_dict[key] = new_difc[:,peak_id]

new_difc_avg = np.mean(new_difc, axis=1)
difc_dict['average'] = new_difc_avg
    
difc_key_order.append('original')
difc_key_order.append('average')

# Put back in Calibration TableWorkspace
difc_col = mtd[cal_wksp].getColumnNames().index('difc')
for detID, difc in zip(detIDs, new_difc_avg):
    mtd[cal_wksp].setCell(detID, difc_col, difc)
    
# Align Components
AlignComponents(CalibrationTable=cal_wksp,
                             Workspace=nom_wksp,
                             ComponentList=[component],
                             XPosition=True,
                             YPosition=True,
                             ZPosition=True
)

# Export new XML for aligned component
ExportGeometry(InputWorkspace=nom_wksp,
                          Components=[component],
                          Filename="NOMAD_bank37_out.xml"
)


# Plotting

plotting=True
if plotting:
    plot_figures(['centre', 'height', 'difc', 'mask'],
                        original_data, 
                        original_peaks,
                        difa_data=original_difa,
                        difc_data=difc_dict,
                         difc_key_order=difc_key_order,
                         tzero_data=original_tzero,
                         mask_data=original_mask,
                         target_peaks=target_dspacing)


plt.show()
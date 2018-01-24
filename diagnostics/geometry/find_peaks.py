#!/usr/bin/env python

from mantid.simpleapi import *

ws = CreateSampleWorkspace(Function="User Defined", UserDefinedFunction="name=LinearBackground, \
   A0=0.3;name=Gaussian, PeakCentre=5, Height=10, Sigma=0.7", NumBanks=1, BankPixelWidth=1, XMin=0, XMax=10, BinWidth=0.1)

table = FindPeaks(InputWorkspace='ws', FWHM='20')

row = table.row(0)

print("Peak 1 {Centre: %.3f, width: %.3f, height: %.3f }" % ( row["centre"],  row["width"], row["height"]))


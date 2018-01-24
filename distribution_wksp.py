#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
from mantid.simpleapi import *

ws_multi = CreateSampleWorkspace("Histogram", "Multiple Peaks")

import matplotlib.pyplot as plt
ws_multi = ConvertToPointData(InputWorkspace=ws_multi)
plt.plot(ws_multi.readX(0), ws_multi.readY(0))
plt.show()
ws_multi = ConvertToHistogram(InputWorkspace=ws_multi)

print("Is the workspace a distribution? " + str(ws_multi.isDistribution()))
print("The workspace has a level background of " +
      str(ws_multi.readY(0)[0]) + " counts.")
print("The largest of which is " +
      str(ws_multi.readY(0)[60]) + " counts." + "\n")

ConvertToDistribution(ws_multi)

print("Is the workspace a distribution? " + str(ws_multi.isDistribution()))
print("The workspace has a level background of " +
      str(ws_multi.readY(0)[0]) + " counts.")
print("The largest of which is " + str(ws_multi.readY(0)[60]) + " counts.")

ws_multi = ConvertToPointData(InputWorkspace=ws_multi)
plt.plot(ws_multi.readX(0), ws_multi.readY(0))
plt.show()
ws_multi = ConvertToHistogram(InputWorkspace=ws_multi)

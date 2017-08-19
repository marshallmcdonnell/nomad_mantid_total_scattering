#/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import os
from mantid.simpleapi import *


diav=0.585
LoadEventNexus(Filename='/SNS/NOM/IPTS-16719/nexus/NOM_82957.nxs.h5', OutputWorkspace='NOM_82957')
CropWorkspace(InputWorkspace='NOM_82957', OutputWorkspace='NOM_82957', XMin='300', XMax='16666.700000000001')
ConvertUnits(InputWorkspace='NOM_82957', OutputWorkspace='NOM_82957', Target='Wavelength')
CropWorkspace(InputWorkspace='NOM_82957', OutputWorkspace='NOM_82957', XMin='0.10000000000000001', XMax='2.8999999999999999')
Rebin(InputWorkspace='NOM_82957', OutputWorkspace='NOM_82957', Params='0.1,0.056,2.9')
SetSample(InputWorkspace='NOM_82957', Geometry={"Center":[0.,0.,0.],"Height":5.,"Radius":diav/2.0,"Shape":"Cylinder"},
              Material={"ChemicalFormula":"V","SampleNumberDensity":0.0721})
MonteCarloAbsorption(InputWorkspace='NOM_82957', OutputWorkspace='absorption_V', EventsPerPoint='3000', Interpolation='CSpline')
SaveNexus(InputWorkspace='absorption_V', Filename=os.path.abspath('.') + '/absorption_V_0_58.nxs')

#!/usr/bin/env python

from mantid.simpleapi import *

LoadEmptyInstrument(Filename="NOMAD_Definition.xml",
                    OutputWorkspace="NOM_geom")
import mantid
filename=mantid.config.getString("defaultsave.directory")+"NOMgeometry.xml"
ExportGeometry(InputWorkspace="NOM_geom",
               Components="bank46,bank47",
               Filename=filename)
import os
if os.path.isfile(filename):
    print("File created: True")

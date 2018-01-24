#!/usr/bin/env python

from mantid.simpleapi import *

ws = LoadEmptyInstrument(Filename="POWGEN_Definition_2015-08-01.xml")
LoadCalFile(InputWorkspace=ws,
      CalFilename="PG3_golden.cal",
      MakeGroupingWorkspace=False,
      MakeOffsetsWorkspace=True,
      MakeMaskWorkspace=True,
      WorkspaceName="PG3")
components="bank25,bank46"
bank25Rot = ws.getInstrument().getComponentByName("bank25").getRotation().getEulerAngles()
bank46Rot = ws.getInstrument().getComponentByName("bank46").getRotation().getEulerAngles()
print("Start bank25 rotation is [{:.3f}.{:.3f},{:.3f}]".format(bank25Rot[0], bank25Rot[1], bank25Rot[2]))
print("Start bank46 rotation is [{:.3f}.{:.3f},{:.3f}]".format(bank46Rot[0], bank46Rot[1], bank46Rot[2]))
AlignComponents(CalibrationTable="PG3_cal",
        Workspace=ws,
        MaskWorkspace="PG3_mask",
        EulerConvention="YZX",
        AlphaRotation=True,
        ComponentList=components)
ws=mtd['ws']
bank25Rot = ws.getInstrument().getComponentByName("bank25").getRotation().getEulerAngles()
bank46Rot = ws.getInstrument().getComponentByName("bank46").getRotation().getEulerAngles()
print("Final bank25 rotation is [{:.3f}.{:.3f},{:.3f}]".format(bank25Rot[0], bank25Rot[1], bank25Rot[2]))
print("Final bank46 rotation is [{:.2f}.{:.3f},{:.3f}]".format(bank46Rot[0], bank46Rot[1], bank46Rot[2]))

#!/usr/bin/env python

from ui.file_load     import ExperimentFileInput
from ui.control_panel import ControlPanel, ControlPanelView, ControlPanelHandler


# Use the ControlPanel to View the Measurement
cp = ControlPanel(experiment_file=ExperimentFileInput())
cp.configure_traits(view=ControlPanelView, handler=ControlPanelHandler)


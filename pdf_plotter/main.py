#!/usr/bin/env python

from ui.file_load     import ExperimentFileInput
from ui.control_panel import ControlPanel
from ui.views         import ControlPanelView
from ui.controllers   import ControlPanelHandler


# Use the ControlPanel to View the Measurement
cp = ControlPanel(experiment_file=ExperimentFileInput())
cp.configure_traits(view=ControlPanelView, handler=ControlPanelHandler)


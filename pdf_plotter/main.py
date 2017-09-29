#!/usr/bin/env python
import numpy as np


from ui.models import Experiment, Measurement, Dataset
from ui.control_panel import ControlPanel
from ui.views import control_panel_view




# Make two datsets w/ titles
x = np.linspace(0,4*np.pi, 200)
y = np.sin(x)
d1 = Dataset(x=x,y=y,title='sin(x)')

y = np.sin(x) * np.sin(x)
d2 = Dataset(x=x,y=y,title='sin(x) * sin(x)')

# Use the datasets to make a Measurement
m1 = Measurement(datasets=[d1, d2], title='sine functions')
#m1.configure_traits(view=measurement_view)

# Now make a second measurement
x = np.linspace(-20,20,200)
m2 = Measurement( datasets = [ Dataset(x=x, y=x*x, title='x^2'),
                               Dataset(x=x, y=x*x*x, title='x^3') ],
                  title='polynomials' )

# Create a Experiment from these two measurements
e1 = Experiment(measurements=[m1,m2],title='Test Functions')

# Use the ControlPanel to View the Measurement
cp = ControlPanel(experiment=e1)
cp.configure_traits(view=control_panel_view)


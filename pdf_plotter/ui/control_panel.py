#!/usr/bin/env python

from traits.api \
    import HasTraits, Instance, Float, Property, Any, on_trait_change, property_depends_on

from matplotlib.figure import Figure

import numpy as np

# Local
from mpl_utilities \
    import MPLFigureEditor, ZoomOnWheel

from models \
    import Experiment, Measurement, Dataset

from views \
    import ControlPanelView

from controllers \
    import ControlPanelHandler


#-----------------------------------------------------------#
# Main Control Panel

class ControlPanel(HasTraits):
    # Passed in measurement
    experiment = Instance(Experiment, ())

    # Figure to display selected dataset
    figure = Instance(Figure, ())

    # The current list of datasets
    datasets  = Property

    # The currently selected dataset
    selected = Any
    
    # The contents of the currently selected dataset
    selected_contents = Property

    # Scale and shift controls
    scale = Float(1.0)
    shift = Float(0.0)

    # Cached plots we keep on plot
    cached_plots = []

    # Limits for min/max for x and y on all datasets to plot (self.selected + self.cached_plots)
    xlim = { 'min' : None, 'max' : None }
    ylim = { 'min' : None, 'max' : None }

    def getLimits(self):
        xlist = []
        ylist = []

        if len(self.cached_plots) > 0:
            xlist = np.append( xlist, [a.x for a in self.cached_plots ] )
            ylist = np.append( ylist, [a.y for a in self.cached_plots ] )

        xlist = np.append( xlist, self.selected.x)
        self.xlim['min'] = min(xlist)
        self.xlim['max'] = max(xlist)

        ylist = np.append( ylist, self.selected.y)
        self.ylim['min'] = min(ylist)
        self.ylim['max'] = max(ylist)

        

    @property_depends_on('experiment')
    def _get_datasets(self):
        datasets = list()
        for measurement in self.experiment.measurements:
            for dataset in measurement.datasets:
                datasets.append(dataset)
        return datasets
            

    @property_depends_on('selected')
    def _get_selected_contents(self):
        if self.selected is None:
            return ''

        return self.selected.x, self.selected.y

    @on_trait_change('selected')
    def plot_selection_and_cache(self):



        if type(self.selected) is Dataset:
            x = self.selected.x
            y = self.selected.y 

            self.figure.pan_zoom = ZoomOnWheel(self.figure)
            axes = self.figure.add_subplot(111)
            self.getLimits()

            axes.set_xlim(self.xlim['min'],self.xlim['max'])
            axes.set_ylim(self.ylim['min'],self.ylim['max'])

            self.plot(axes, x, y)
    

    @on_trait_change('scale,shift')
    def plot_modification(self):
        try:
            scale = self.scale
            shift = self.shift

            x = self.selected.x
            y = scale*(self.selected.y) + shift 

            axes = self.figure.add_subplot(111)

            self.plot(axes, x, y)
        except AttributeError:
            pass
 
    def plot(self, axes, x, y):

        if not axes.lines:
            axes.plot(x, y)
        else:
            l = axes.lines[0]
            l.set_xdata(x)
            l.set_ydata(y)

        for cached_plot in self.cached_plots:
            axes.plot(cached_plot.x, cached_plot.y)

        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()


if __name__ == "__main__":

    # Make two datsets w/ titles
    x = np.linspace(0,4*np.pi, 200)
    y = np.sin(x)
    d1 = Dataset(x=x,y=y,title='sin(x)') 

    y = np.sin(x) * np.sin(x)
    d2 = Dataset(x=x,y=y,title='sin(x) * sin(x)') 

    # Use the datasets to make a Measurement
    m1 = Measurement(datasets=[d1, d2], title='sine functions')

    # Now make a second measurement
    x = np.linspace(-20,20,200)
    m2 = Measurement( datasets = [ Dataset(x=x, y=x*x, title='x^2'),
                                   Dataset(x=x, y=x*x*x, title='x^3') ],
                      title='polynomials' )

    # Create a Experiment from these two measurements
    e1 = Experiment(measurements=[m1,m2],title='Test Functions')

    # Use the ControlPanel to View the Measurement
    cp = ControlPanel(experiment=e1)
    cp.configure_traits(view=ControlPanelView, handler=ControlPanelHandler)

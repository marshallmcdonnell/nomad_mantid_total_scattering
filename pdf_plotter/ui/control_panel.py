#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from traits.api \
    import HasTraits, Instance, Str, List, Float, Property, Any, \
           on_trait_change, property_depends_on

from traitsui.api \
    import CheckListEditor

from matplotlib import cm
from matplotlib.figure import Figure

import numpy as np

# Local
from mpl_utilities \
    import MPLFigureEditor, ZoomOnWheel, DraggableLegend
 
from models \
    import Experiment, Measurement, Dataset

from views \
    import ControlPanelView

from controllers \
    import ControlPanelHandler


#-----------------------------------------------------------#
# Main Control Panel

class ControlPanel(HasTraits):
    #-------------------------------------------------------#
    # Traits

    # Passed in measurement
    experiment = Instance(Experiment, ())

    # Figure to display selected dataset and axes for figure
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
    cached_plots = List

    # List of color maps available
    cmap_list = List(sorted([ cmap for cmap in cm.datad if not cmap.endswith("_r") ], key=lambda s: s.upper()))

    # Selected color map
    selected_cmap = Any

    # Selected color map  contents
    selected_cmap_contents = Property

    # Limits for min/max for x and y on all datasets to plot (self.selected + self.cached_plots)
    xlim = { 'min' : None, 'max' : None }
    ylim = { 'min' : None, 'max' : None }

    #-------------------------------------------------------#
    # Utilities

    def __setupColorMap(self,num_lines=8,reset_cycler=True):
        if num_lines < len(self.cached_plots):
            num_lines = len(self.cached_plots)
        if reset_cycler:  
            self.figure.gca().set_prop_cycle(None)
        myCMap = cm.get_cmap(name=self.selected_cmap_contents)
        cm_subsection = np.linspace( 0.0, 1.0, num_lines )
        self.colors = [ myCMap(x) for x in cm_subsection ]

    def addPlotToNode(self,dataset):
        if 'Other' not in [ m.title for m in self.experiment.measurements ]:
            other = Measurement(datasets=[dataset], title='Other')
            self.experiment.measurements.append(other)
        else:
            other = [ m for m in self.experiment.measurements if m.title == 'Other' ]
            if len(other) != 1:
                print("ERROR: Found more than 1 'Other' Measurement nodes. Resuming...")
            other = other[0]
            other.datasets.append(dataset)

        
        


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

    def plot(self, axes, x, y,title):
        if not axes.lines:
            axes.plot(x, y,'bo',label=title)

        else:
            l = axes.lines[0]
            l.set_data(x,y)
            l.set_label(title)
            
            self.__setupColorMap()
            for i, (cached_plot, color) in enumerate(zip(self.cached_plots,self.colors)):
                l = axes.lines[i+1]
                l.set_data(cached_plot.x,cached_plot.y)
                l.set_color(color)
                l.set_label(cached_plot.title)

        leg = axes.legend()
        leg = DraggableLegend(leg)
            

        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()

    #-------------------------------------------------------#
    # Dynamic

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


    @property_depends_on('selected_cmap')
    def _get_selected_cmap_contents(self):
        if self.selected_cmap:
            return self.selected_cmap[0]
        return 'Set1'

    @on_trait_change('selected,selected_cmap')
    def plot_selection(self):
        if len(self.figure.axes) == 0:
            self.figure.pan_zoom = ZoomOnWheel(self.figure)
            self.axes = self.figure.add_subplot(111)

        if type(self.selected) is Dataset:
            x = self.selected.x
            y = self.selected.y 

            self.getLimits()

            self.axes.set_xlim(self.xlim['min'],self.xlim['max'])
            self.axes.set_ylim(self.ylim['min'],self.ylim['max'])

            self.plot(self.axes, x, y,self.selected.title)

    @on_trait_change('scale,shift')
    def plot_modification(self):
        try:
            scale = self.scale
            shift = self.shift

            x = self.selected.x
            y = scale*(self.selected.y) + shift

            self.plot(self.axes, x, y, self.selected.title)
        except AttributeError:
            pass
 
if __name__ == "__main__":

    # Make datsets w/ titles
    x = np.linspace(0,4*np.pi, 200)
    y = np.sin(x)
    d1 = Dataset(x=x,y=y,title='sin(x)') 

    y = np.sin(x) * np.sin(x)
    d2 = Dataset(x=x,y=y,title='sin(x) * sin(x)') 

    y = np.cos(x)
    d3 = Dataset(x=x,y=y,title='cos(x)')

    y = np.cos(x) * np.cos(x)
    d4 = Dataset(x=x,y=y,title='cos(x) * cos(x)')

    y = np.sin(x) * np.cos(x)
    d5 = Dataset(x=x,y=y,title='sin(x) * cos(x)')

    # Use the datasets to make a Measurement
    m1 = Measurement(datasets=[d1, d2, d3, d4, d5], title='sine functions')

    # Now make a second measurement
    x = np.linspace(-2,2,200)
    m2 = Measurement( datasets = [ Dataset(x=x, y=x*x, title='x^2'),
                                   Dataset(x=x, y=x*x*x, title='x^3') ],
                      title='polynomials' )

    # Create a Experiment from these two measurements
    e1 = Experiment(measurements=[m1,m2],title='Test Functions')

    # Use the ControlPanel to View the Measurement
    cp = ControlPanel(experiment=e1)
    cp.configure_traits(view=ControlPanelView, handler=ControlPanelHandler)

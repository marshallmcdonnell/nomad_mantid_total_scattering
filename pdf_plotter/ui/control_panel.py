#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from traits.api \
    import HasTraits, Instance, List, Float, Property, Any, \
    on_trait_change, property_depends_on

from matplotlib import cm
from matplotlib.figure import Figure

import numpy as np

# Local
from mpl_utilities \
    import ZoomOnWheel, DraggableLegend

from models \
    import Experiment, Measurement, Dataset

from views \
    import SofqPlotView, ControlsView, ControlPanelView

from controllers \
    import ControlPanelHandler

# -----------------------------------------------------------#
# Figure View


class SofqPlot(HasTraits):
    # View
    view = SofqPlotView

    # Figure to display selected dataset and axes for figure
    figure = Instance(Figure, ())


# -----------------------------------------------------------#
# Controls View

class Controls(HasTraits):
    # View
    view = ControlsView

    # -------------------------------------------------------#
    # Traits

    # Passed in measurement
    experiment = Instance(Experiment, ())

    # The current list of datasets
    datasets = Property

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
    cmap_list = List(sorted(
        [cmap for cmap in cm.datad if not cmap.endswith("_r")],
        key=lambda s: s.upper()
    )
    )

    # Selected color map
    selected_cmap = Any

    # Selected color map  contents
    selected_cmap_contents = Property

    # Limits for min/max for x and y on all datasets to plot (self.selected +
    # self.cached_plots)
    xlim = {'min': None, 'max': None}
    ylim = {'min': None, 'max': None}

    # -------------------------------------------------------#
    # Utilities

    def getLimits(self):
        xlist = []
        ylist = []

        if len(self.cached_plots) > 0:
            xlist = np.append(xlist, [a.x for a in self.cached_plots])
            ylist = np.append(ylist, [a.y for a in self.cached_plots])

        xlist = np.append(xlist, self.selected.x)
        self.xlim['min'] = min(xlist)
        self.xlim['max'] = max(xlist)

        ylist = np.append(ylist, self.selected.y)
        self.ylim['min'] = min(ylist)
        self.ylim['max'] = max(ylist)

    def addPlotToNode(self, dataset):
        if 'Other' not in [m.title for m in self.experiment.measurements]:
            other = Measurement(datasets=[dataset], title='Other')
            self.experiment.measurements.append(other)
        else:
            other = [
                m for m in self.experiment.measurements if m.title == 'Other']
            if len(other) != 1:
                print("ERROR: More than 1 'Other' Measurement. Resuming...")
            other = other[0]
            other.datasets.append(dataset)

    # -------------------------------------------------------#
    # Dynamic

    # Gives the X,Y of the selected node and stores in selected_contents
    @property_depends_on('selected')
    def _get_selected_contents(self):
        if self.selected is None:
            return ''
        return self.selected.x, self.selected.y

    # Extracts Datasets models that are stored in the Experiment model
    @property_depends_on('experiment')
    def _get_datasets(self):
        datasets = list()
        for measurement in self.experiment.measurements:
            for dataset in measurement.datasets:
                datasets.append(dataset)
        return datasets

    # Gets the
    @property_depends_on('selected_cmap')
    def _get_selected_cmap_contents(self):
        if self.selected_cmap:
            return self.selected_cmap[0]
        return 'Set1'


# -----------------------------------------------------------#
# Main Control Panel

class ControlPanel(HasTraits):

    # -------------------------------------------------------#
    # Traits

    # S(Q) Plot
    sofq_plot = Instance(SofqPlot, ())

    # Controls for adjusting plots
    controls = Instance(Controls)

    # Current colors of contents that is setup by __setupColorMap
    _colors = list()

    # -------------------------------------------------------#
    # Utilities

    # Initialize the ColorMap selected for all the cached plots
    def __setupColorMap(self, num_lines=8, reset_cycler=True):
        # If there are more lines than
        if num_lines < len(self.controls.cached_plots):
            num_lines = len(self.controls.cached_plots)

        # Reset the cycler so we don't get the same colors in order of the
        # ColorMap
        if reset_cycler:
            self.sofq_plot.figure.gca().set_prop_cycle(None)

        # Get the currently selected ColorMap from the Controls
        myCMap = cm.get_cmap(name=self.controls.selected_cmap_contents)

        # Create a value for each line that corresponds to the color in the
        # ColorMap, ranged from 0->1
        cm_subsection = np.linspace(0.0, 1.0, num_lines)

        # Pull the colors from the ColorMap selected using the values created
        # in cm_subsection
        self._colors = [myCMap(x) for x in cm_subsection]

    # Plots the X, Y data (cached and selected) on the given Axes and re-draws
    # the canvas
    def plot(self, axes, x, y, title):
        # Add first line to plot if none already
        if not axes.lines:
            axes.plot(x, y, 'bo', label=title)

        # Adjust the current X, Y data of the selected line the plot
        else:
            l = axes.lines[0]
            l.set_data(x, y)
            l.set_label(title)

            # Setup the ColorMap for the cache plots
            self.__setupColorMap()

            # Loop over lines in the Axes for the cached plots and modify
            # accordingly
            for i, (cached_plot, color) in enumerate(
                    zip(self.controls.cached_plots, self._colors)):
                l = axes.lines[i + 1]
                l.set_data(cached_plot.x, cached_plot.y)
                l.set_color(color)
                l.set_label(cached_plot.title)

        # Make the Legend: draggable, wheel-adjusted font, right-click hidable,
        leg = axes.legend()
        leg = DraggableLegend(leg)

        # Re-draw the Canvas of the Figure with our changes
        canvas = self.sofq_plot.figure.canvas
        if canvas is not None:
            canvas.draw()

    def _get_axes(self):
        # If no Axes for the figure, 1) Make the figure zoomable by mouse wheel
        # and 2) initialize an Axes
        if len(self.sofq_plot.figure.axes) == 0:
            self.sofq_plot.figure.pan_zoom = ZoomOnWheel(self.sofq_plot.figure)
            axes = self.sofq_plot.figure.add_subplot(111)

        # Get only Axes currently available
        else:
            axes = self.sofq_plot.figure.axes[0]

        return axes

    # -------------------------------------------------------#
    # Dynamic

    # Re-plot when we either select another Dataset or if we change the
    # ColorMap
    @on_trait_change('controls.selected,controls.selected_cmap')
    def plot_selection(self):

        # Only if the Experiment Node is of a Dataset-type, get the contents
        # and plot
        if isinstance(self.controls.selected, Dataset):

            # Reset the scale and shift
            self.controls.scale = 1.0
            self.controls.shift = 0.0

            # Get the Axes
            axes = self._get_axes()

            # Pull the X, Y from the selected Dataset
            x = self.controls.selected.x
            y = self.controls.selected.y

            # Get the X, Y limits from all plots (selected + cached)
            self.controls.getLimits()

            # Set the limits
            axes.set_xlim(self.controls.xlim['min'], self.controls.xlim['max'])
            axes.set_ylim(self.controls.ylim['min'], self.controls.ylim['max'])

            # Plot / Re-plot
            self.plot(axes, x, y, self.controls.selected.title)

    @on_trait_change('controls.scale,controls.shift')
    def plot_modification(self):
        try:
            axes = self._get_axes()
            scale = self.controls.scale
            shift = self.controls.shift

            x = self.controls.selected.x
            y = scale * (self.controls.selected.y) + shift

            self.plot(axes, x, y, self.controls.selected.title)
        except AttributeError:
            pass


if __name__ == "__main__":

    # Make datsets w/ titles
    x = np.linspace(0, 4 * np.pi, 200)
    y = np.sin(x)
    d1 = Dataset(x=x, y=y, title='sin(x)')

    y = np.sin(x) * np.sin(x)
    d2 = Dataset(x=x, y=y, title='sin(x) * sin(x)')

    y = np.cos(x)
    d3 = Dataset(x=x, y=y, title='cos(x)')

    y = np.cos(x) * np.cos(x)
    d4 = Dataset(x=x, y=y, title='cos(x) * cos(x)')

    y = np.sin(x) * np.cos(x)
    d5 = Dataset(x=x, y=y, title='sin(x) * cos(x)')

    # Use the datasets to make a Measurement
    m1 = Measurement(datasets=[d1, d2, d3, d4, d5], title='sine functions')

    # Now make a second measurement
    x = np.linspace(-2, 2, 200)
    m2 = Measurement(datasets=[Dataset(x=x, y=x * x, title='x^2'),
                               Dataset(x=x, y=x * x * x, title='x^3')],
                     title='polynomials')

    # Create a Experiment from these two measurements
    e1 = Experiment(measurements=[m1, m2], title='Test Functions')

    # Use the ControlPanel to View the Measurement
    cp = ControlPanel(controls=Controls(experiment=e1))
    cp.configure_traits(view=ControlPanelView, handler=ControlPanelHandler)

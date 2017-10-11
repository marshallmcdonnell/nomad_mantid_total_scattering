#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import argparse
import numpy as np

# Traits
from traits.api \
    import HasTraits, Instance, Property, CFloat, Event,\
    on_trait_change

# Matplotlib
from matplotlib import cm
from matplotlib.figure import Figure


# Local
from mpl_utilities \
    import ZoomOnWheel, DraggableLegend

import models
import views 
import controls 
import controllers 
import file_load 

# -----------------------------------------------------------#
# Figure Model


class SofqPlot(HasTraits):
    # View
    view = views.SofqPlotView

    # Figure to display selected dataset and axes for figure
    figure = Instance(Figure, ())


# -----------------------------------------------------------#
# Main Control Panel

class ControlPanel(HasTraits):

    # -------------------------------------------------------#
    # Traits

    experiment_file = Instance(file_load.ExperimentFileInput)

    # S(Q) Plot
    sofq_plot = Instance(SofqPlot, ())

    # Controls for adjusting plots
    controls = Instance(controls.Controls)

    # Status
    load_status = Property(depends_on='experiment_file.load_status')

    # Selected node
    selected = Property(depends_on='controls.selected')

    # Button event fired
    button_pressed = Property(depends_on='controls.node_buttons.button_event')

    # Current colors of contents that is setup by _setupColorMap
    _colors = list()

    # Stuff for plots
    cache_start_index = 0
    dataset = Instance(models.Dataset)
    corrected_datasets = Instance(models.CorrectedDatasets)

    plot_xmin = CFloat
    plot_xmax = CFloat
    plot_ymin = CFloat
    plot_ymax = CFloat

    # -------------------------------------------------------#
    # Utilities

    # Updates the displayed load status based on notifications from Controls
    def _get_load_status(self):
        return ("%s" % self.experiment_file.load_status)

    # Updates the displayed load status based on notifications from Controls
    def _get_selected(self):
        return self.controls.selected

    # Updates the buttons fired from Controls
    def _get_button_pressed(self):
        return self.controls.node_buttons

    # Initialize the ColorMap selected for the list of plots
    def _setupColorMap(self, plot_list, num_lines=8, reset_cycler=True):
        # If there are more lines than
        if num_lines < len(plot_list):
            num_lines = len(plot_list)

        # Reset the cycler so we don't get the same colors in order of the
        # ColorMap
        if reset_cycler:
            self.sofq_plot.figure.gca().set_prop_cycle(None)

        # Get the currently selected ColorMap from the Controls
        myCMap = cm.get_cmap(
            name=self.controls.node_controls.selected_cmap_contents)

        # Create a value for each line that corresponds to the color in the
        # ColorMap, ranged from 0->1
        cm_subsection = np.linspace(0.0, 1.0, num_lines)

        # Pull the colors from the ColorMap selected using the values created
        # in cm_subsection
        self._colors = [myCMap(x) for x in cm_subsection]

    # Sets the limits for the plots in the figure (cached and selected)
    def _get_limits_on_plot(self, xin, yin):

        # Apply x-range filter
        x, y = self.controls.node_controls.filter_xrange(xin, yin)

        xlist = list()
        ylist = list()

        if len(self.controls.cached_plots) > 0:
            for a in self.controls.cached_plots:

                xlist = np.append(xlist, a.x, axis=None)
                ylist = np.append(ylist, a.y, axis=None)

        xlist = np.append(xlist, x)
        xlist = np.append(xlist, self.controls.node_controls.xmin)
        xlist = np.append(xlist, self.controls.node_controls.xmax)

        ylist = np.append(ylist, y)

        ylist[ylist == np.inf] = 0.0
        ylist[ylist == -np.inf] = 0.0
        ylist[np.isnan(ylist)] = 0.0

        self.plot_xmin = min(xlist)
        self.plot_xmax = max(xlist)
        self.plot_ymin = min(ylist)
        self.plot_ymax = max(ylist)

    # Add the cached lines back to the plot (style taken care of in plot_cached)
    def add_cached(self):
        axes = self.get_axes()
        for cached_plot in self.controls.cached_plots:
            axes.plot(cached_plot.x, cached_plot.y)

    # Loop over lines in the Axes for the cached plots and modify
    # accordingly
    def plot_cached(self):
        axes = self.get_axes()
        for i, (cached_plot, color) in enumerate(
                zip(self.controls.cached_plots, self._colors)):
            l = axes.lines[i + self.cache_start_index]
            l.set_data(cached_plot.x, cached_plot.y)
            l.set_color(color)
            l.set_marker(None)
            l.set_linestyle('-')
            l.set_label(cached_plot.title)

    # Clear the plot
    def clear_plot(self):
        # Get the Axes
        axes = self.get_axes()
        axes.cla()

    # If selected Tree Node is Dataset, plot the Dataset
    def plot_dataset(self):
        # Get the Axes
        axes = self.get_axes()

        # Reset the scale and shift
        self.controls.node_controls.scale_factor = 1.0
        self.controls.node_controls.shift_factor = 0.0

        # Get X, Y for selected Dataset
        x = self.controls.selected.x
        y = self.controls.selected.y

        # Get the limits
        self._get_limits_on_plot(x, y)

        # Set the limits
        axes.set_xlim(self.plot_xmin, self.plot_xmax)
        axes.set_ylim(self.plot_ymin, self.plot_ymax)

        # Use the modifications to adjust the x, y line
        self.set_xy(axes, x, y, self.controls.selected.title)

        # Set index to start plotting cached plots
        self.cache_start_index = 1

    # If selected Tree Node is CorrectedDatasets, plot all Dataset children
    def plot_corrected_datasets(self):
        # Get the Datasets
        datasets = self.controls.selected.datasets

        # Get the Axes
        axes = self.get_axes()

        for i, dataset in enumerate(datasets):
            axes.plot(dataset.x, dataset.y, label=dataset.title)

        # Set index to start plotting cached plots
        self.cache_start_index = len(datasets)

    # Plots the X, Y data (cached and selected) on the given Axes and re-draws
    # the canvas
    def set_xy(self, axes, x, y, title):

        # Add first line to plot if none already
        if not axes.lines:
            axes.plot(x, y, label=title)

        # Plot xy
        l = axes.lines[0]
        l.set_data(x, y)
        l.set_color('b')
        l.set_marker('o')
        l.set_markersize(4.0)
        l.set_linestyle('--')
        l.set_label(title)

    def get_axes(self):
        # If no Axes for the figure, 1) Make the figure zoomable by mouse wheel
        # and 2) initialize an Axes
        if len(self.sofq_plot.figure.axes) == 0:
            self.sofq_plot.figure.pan_zoom = ZoomOnWheel(self.sofq_plot.figure)
            axes = self.sofq_plot.figure.add_subplot(111)

        # Get only Axes currently available
        else:
            axes = self.sofq_plot.figure.axes[0]

        return axes

    def redraw_canvas(self):
        axes = self.get_axes()

        # Make the Legend: draggable, wheel-adjusted font, right-click hidable,
        leg = axes.legend()
        leg = DraggableLegend(leg)

        # Re-draw the Canvas of the Figure with our changes
        canvas = self.sofq_plot.figure.canvas
        if canvas is not None:
            canvas.draw()

    # -------------------------------------------------------#
    # Dynamic

    @on_trait_change('experiment_file.experiment')
    def experiment_updated(self):
        experiment = self.experiment_file.experiment
        if experiment:
            if self.controls:
                self.controls.experiment = experiment
            else:
                self.controls = controls.Controls(experiment=experiment,
                                         node_controls=controls.DatasetNodeControls(),
                                         node_buttons=controls.DatasetNodeButtons())

    @on_trait_change('experiment_file.load_status')
    def update_status(self):
        self.status = self.experiment_file.load_status

    # Re-plot when we either select another Dataset or if we change the
    # ColorMap
    @on_trait_change('controls.selected,controls.node_controls.selected_cmap')
    def plot_selection(self):
        try:
            # Plot the selected Tree Node based on its type
            if isinstance(self.controls.selected, models.Dataset):
                self.clear_plot()
                self.plot_dataset()

            elif isinstance(self.controls.selected, models.CorrectedDatasets):
                self.clear_plot()
                self.plot_corrected_datasets()

            # Add cached lines back to plot
            self.add_cached()

            # Setup the ColorMap for the cache plots
            self._setupColorMap(self.controls.cached_plots)

            # Plot the cached lines
            self.plot_cached()

            # Redraw the canvas of the figure
            self.redraw_canvas()

        except AttributeError:
            pass

    # Re-plot when we apply a shift or scale factor
    @on_trait_change('controls.node_controls.scale_factor,'
                     'controls.node_controls.shift_factor,'
                     'controls.node_controls.xmin,'
                     'controls.node_controls.xmax,')
    def plot_dataset_modification(self):
        try:
            axes = self.get_axes()
            scale = self.controls.node_controls.scale_factor
            shift = self.controls.node_controls.shift_factor

            x = self.controls.selected.x
            y = scale * (self.controls.selected.y) + shift

            # Apply x-range filter
            x, y = self.controls.node_controls.filter_xrange(x, y)

            # Get the X, Y limits from all plots (selected + cached)
            self._get_limits_on_plot(x, y)

            # Set the limits
            axes.set_xlim(self.plot_xmin, self.plot_xmax)
            axes.set_ylim(self.plot_ymin, self.plot_ymax)

            # Use the modifications to adjust the x, y line
            self.set_xy(axes, x, y, self.controls.selected.title)

            # Redraw the canvas of the figure
            self.redraw_canvas()

        except AttributeError:
            pass


if __name__ == "__main__":
    cp = ControlPanel(experiment_file=file_load.ExperimentFileInput())
    cp.configure_traits(view=views.ControlPanelView, handler=controllers.ControlPanelHandler)

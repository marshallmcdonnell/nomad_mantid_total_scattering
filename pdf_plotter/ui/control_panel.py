#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import argparse
import numpy as np

# Traits
from traits.api \
    import HasTraits, Instance, Property,  \
    on_trait_change

# Matplotlib
from matplotlib import cm
from matplotlib.figure import Figure


# Local
from mpl_utilities \
    import ZoomOnWheel, DraggableLegend

from models \
    import Experiment, Measurement, CorrectedDatasets, Dataset

from views \
    import SofqPlotView, ControlPanelView

from controls \
    import DatasetNodeControls, Controls

from controllers \
    import ControlPanelHandler

from file_load \
    import ExperimentFileInput

# -----------------------------------------------------------#
# Figure Model


class SofqPlot(HasTraits):
    # View
    view = SofqPlotView

    # Figure to display selected dataset and axes for figure
    figure = Instance(Figure, ())


# -----------------------------------------------------------#
# Main Control Panel

class ControlPanel(HasTraits):

    # -------------------------------------------------------#
    # Traits

    experiment_file = Instance(ExperimentFileInput)

    # S(Q) Plot
    sofq_plot = Instance(SofqPlot, ())

    # Controls for adjusting plots
    controls = Instance(Controls)

    # Status
    load_status = Property(depends_on='experiment_file.load_status')

    # Status
    selected = Property(depends_on='controls.selected')

    # Current colors of contents that is setup by _setupColorMap
    _colors = list()

    # Stuff for plots
    cache_start_index = 0
    dataset = Instance(Dataset)
    corrected_datasets = Instance(CorrectedDatasets)

    # -------------------------------------------------------#
    # Utilities

    # Updates the displayed load status based on notifications from Controls
    def _get_load_status(self):
        return ("%s" % self.experiment_file.load_status)

    # Updates the displayed load status based on notifications from Controls
    def _get_selected(self):
        return self.controls.selected

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
        myCMap = cm.get_cmap(name=self.controls.node_controls.selected_cmap_contents)

        # Create a value for each line that corresponds to the color in the
        # ColorMap, ranged from 0->1
        cm_subsection = np.linspace(0.0, 1.0, num_lines)

        # Pull the colors from the ColorMap selected using the values created
        # in cm_subsection
        self._colors = [myCMap(x) for x in cm_subsection]

    # Use Controls X-range to select subset of the domain of the plot
    def _filter_xrange(self, xset, yset):
        xmin = self.controls.node_controls.xmin
        xmax = self.controls.node_controls.xmax

        xout = list()
        yout = list()
        for x, y in zip(xset, yset):
            if xmin <= x and x <= xmax:
                xout.append(x)
                yout.append(y)

        return xout, yout

    # Add the cached lines back the plot (style taken care of in plot_cached)
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

        # Pull the X, Y from the selected Dataset
        x = self.controls.selected.x
        y = self.controls.selected.y

        # Apply x-range filter
        x, y = self._filter_xrange(x, y)

        # Get the X, Y limits from all plots (selected + cached)
        self.controls.getLimitsOnPlot(x, y)

        # Set the limits
        axes.set_xlim(
            self.controls.xlim_on_plot['min'],
            self.controls.xlim_on_plot['max'])
        axes.set_ylim(
            self.controls.ylim_on_plot['min'],
            self.controls.ylim_on_plot['max'])

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
            x = dataset.x[:-1]
            y = dataset.y
            axes.plot(x, y, label=dataset.title)

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
                self.controls = Controls(experiment=experiment,
                                         node_controls=DatasetNodeControls())

    @on_trait_change('experiment_file.load_status')
    def update_status(self):
        self.status = self.experiment_file.load_status

    # Re-plot when we either select another Dataset or if we change the
    # ColorMap
    @on_trait_change('controls.selected,controls.node_controls.selected_cmap')
    def plot_selection(self):
        try:
            # Plot the selected Tree Node based on its type
            if isinstance(self.controls.selected, Dataset):
                self.clear_plot()
                self.plot_dataset()

            elif isinstance(self.controls.selected, CorrectedDatasets):
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
            x, y = self._filter_xrange(x, y)

            # Get the X, Y limits from all plots (selected + cached)
            self.controls.getLimitsOnPlot(x, y)

            # Set the limits
            axes.set_xlim(
                self.controls.xlim_on_plot['min'],
                self.controls.xlim_on_plot['max'])
            axes.set_ylim(
                self.controls.ylim_on_plot['min'],
                self.controls.ylim_on_plot['max'])

            # Use the modifications to adjust the x, y line
            self.set_xy(axes, x, y, self.controls.selected.title)

            # Redraw the canvas of the figure
            self.redraw_canvas()

        except AttributeError:
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy_data', action='store_true',
                        help="Loads Experiment Tree w/ dummy data.")
    parser.add_argument(
        '--controls_view_only',
        action='store_true',
        help="Loads Experiment Tree w/ dummy data and just view Controls.")
    args = parser.parse_args()

    if args.dummy_data:

        # Make datasets w/ titles
        x = np.linspace(0, 4 * np.pi, 200)
        y = 1.21 * np.sin(x)
        d1 = Dataset(x=x, y=y, title='Bank 1', info={'correction': 'S/V'})

        y = 1.24 * np.sin(x)
        d2 = Dataset(x=x, y=y, title='Bank 2', info={'correction': 'S/V'})

        y = 0.90 * np.cos(x)
        d3 = Dataset(x=x, y=y, title='Bank 1', info={'correction': '(S-C)/V'})

        y = 1.20 * np.cos(x)
        d4 = Dataset(x=x, y=y, title='Bank 2', info={'correction': '(S-C)/V'})

        y = 1.21 * np.cos(x)
        d5 = Dataset(x=x, y=y, title='Bank 3', info={'correction': '(S-C/V'})

        # Use the datasets to make a CorrectedDatasets
        cd1 = CorrectedDatasets(datasets=[d1, d2], title='S/V')
        cd2 = CorrectedDatasets(datasets=[d3, d4, d5], title='(S-C)/V')

        # Combine the CorrectedDatasets to make a Measurment for the sample
        m1 = Measurement(corrected_datasets=[cd1, cd2], title='Sample')

        # Now make a second measurement for the container
        x = np.linspace(-2, 2, 200)
        m2 = Measurement(
            corrected_datasets=[
                CorrectedDatasets(datasets=[
                    Dataset(x=x,
                            y=0.80 * x * x,
                            title='Bank 1',
                            info={'correction': '(C-CB)/V'},
                            ),
                    Dataset(x=x,
                            y=0.97 * x * x,
                            title='Bank 2',
                            info={'correction': '(C-CB)/V'},
                            ),
                ],
                    title='(C-CB)/V',
                ),
                CorrectedDatasets(datasets=[
                    Dataset(x=x,
                            y=0.80 * x * x * x,
                            title='Bank 1',
                            info={'correction': '1/A(C-CB)/V - MS'},
                            ),
                    Dataset(x=x,
                            y=0.97 * x * x * x,
                            title='Bank 2',
                            info={'correction': '1/A(C-CB)/V - MS'},
                            ),
                ],
                    title="1/A(C-CB)/V - MS",
                ),
            ],
            title='Container',
        )

        # Create a Experiment from these two measurements
        e1 = Experiment(measurements=[m1, m2], title='Si_NOM97884')

        # Use the ControlPanel to View the Measurement
        if args.controls_view_only:
            c = Controls(experiment=e1)
            c.configure_traits()
        else:
            cp = ControlPanel(experiment_file=ExperimentFileInput(),
                              controls=Controls(experiment=e1))
            cp.configure_traits(
                view=ControlPanelView,
                handler=ControlPanelHandler)

    else:
        cp = ControlPanel(experiment_file=ExperimentFileInput())
        cp.configure_traits(view=ControlPanelView, handler=ControlPanelHandler)

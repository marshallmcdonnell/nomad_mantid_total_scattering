#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import os
import time
import argparse
import numpy as np

# Traits
from traits.api \
    import HasTraits, Instance, List, CFloat, Property, Any, \
    Str, Button, on_trait_change, property_depends_on

from traitsui.file_dialog \
    import open_file, FileInfo

# Matplotlib
from matplotlib import cm
from matplotlib.figure import Figure


# Local
from mpl_utilities \
    import ZoomOnWheel, DraggableLegend

from models \
    import Experiment, Measurement, CorrectedDatasets, Dataset

from views \
    import SofqPlotView, ControlsView, ControlPanelView, \
    ExperimentFileInputView

from controllers \
    import ControlPanelHandler

from thread_workers \
    import NexusFileThread, ExperimentThread

# -----------------------------------------------------------#
# Figure Model


class SofqPlot(HasTraits):
    # View
    view = SofqPlotView

    # Figure to display selected dataset and axes for figure
    figure = Instance(Figure, ())

# -----------------------------------------------------------#
# Experiment File Input Model


class ExperimentFileInput(HasTraits):
    # View
    view = ExperimentFileInputView

    # Load button
    load_button = Button("Load Experiment...")

    # NeXus filename that is loaded in
    filename = Str

    # Thread to handle loading in the file
    file_thread = Instance(NexusFileThread)

    # Thread to handle putting together the Experiment from CorrectedDatasets
    experiment_thread = Instance(ExperimentThread)

    # status
    load_status = Str('Load in a file.')

    # Dataset dictionary: key=title, value={CorrectedDatasets, tag}
    corrected_datasets = dict()

    # Returned Experiment
    experiment = Instance(Experiment)

    # Get update from thread
    def update_status(self, status):
        self.load_status = status

    # Get experiment update from thread
    def update_experiment(self, experiment):
        self.experiment = experiment

    # Handle the user clicking the Load button
    def _load_button_changed(self):
        f = open_file(file_name=os.getcwd(),
                      extensions=[FileInfo()],
                      filter=['*.nxs', '*.dat'])
        self.filename = f
        if f != '':
            name, ext = os.path.splitext(f)
            if ext == '.nxs':
                self.load_and_extract_nexus(f)
                while self.file_thread.isAlive():
                    time.sleep(1)
                self.form_experiment()
            elif ext == '.dat':
                self.load_and_extract_dat_file(f)

    # Parse the Experiment NeXus file
    def load_and_extract_nexus(self, f):
        self.file_thread = NexusFileThread(f)
        self.file_thread.update_status = self.update_status
        self.file_thread.corrected_datasets = self.corrected_datasets
        self.file_thread.start()

    def form_experiment(self):
        self.experiment_thread = ExperimentThread()
        self.experiment_thread.update_status = self.update_status
        self.experiment_thread.update_experiment = self.update_experiment
        self.experiment_thread.corrected_datasets = self.corrected_datasets
        self.experiment_thread.filename = self.filename
        self.experiment_thread.experiment = self.experiment
        self.experiment_thread.start()


# -----------------------------------------------------------#
# Controls Model


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

    # Scale controls
    scale_min = CFloat(0.5)
    scale_max = CFloat(1.5)
    scale_factor = CFloat(1.0)

    # Scale controls
    shift_min = CFloat(-5.0)
    shift_max = CFloat(5.0)
    shift_factor = CFloat(0.0)

    # X-range controls
    xmin = CFloat(0.0)
    xmin_min = CFloat(0.0)
    xmin_max = CFloat(5.0)

    xmax = CFloat(40.0)
    xmax_min = CFloat(0.0)
    xmax_max = CFloat(2.0)

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

    # Limits for min/max for x and y on all datasets in experiment
    xlim_for_exp = {'min': None, 'max': None}
    ylim_for_exp = {'min': None, 'max': None}

    # Limits for min/max for x and y on all datasets to plot (self.selected +
    # self.cached_plots)
    xlim_on_plot = {'min': None, 'max': None}
    ylim_on_plot = {'min': None, 'max': None}

    # -------------------------------------------------------#
    # Utilities

    # Sets the limits for the X-range axis using all datasets
    def getLimitsForExperiment(self):
        xlist = []
        ylist = []
        for d in self.datasets:
            xlist = np.append(xlist, d.x, axis=None)
            ylist = np.append(ylist, d.y, axis=None)

        try:
            self.xlim_for_exp['min'] = min(xlist)
            self.xlim_for_exp['max'] = max(xlist)
            self.ylim_for_exp['min'] = min(ylist)
            self.ylim_for_exp['max'] = max(ylist)

        except ValueError:
            return

    # Sets the limits for the plots in the figure (cached and selected)
    def getLimitsOnPlot(self, xin, yin):
        xlist = []
        ylist = []

        if len(self.cached_plots) > 0:
            for a in self.cached_plots:

                xlist = np.append(xlist, a.x, axis=None)
                ylist = np.append(ylist, a.y, axis=None)

        xlist = np.append(xlist, xin)
        xlist = np.append(xlist, self.xmin)
        xlist = np.append(xlist, self.xmax)

        ylist = np.append(ylist, yin)

        ylist[ ylist ==  np.inf ] = 0.0
        ylist[ ylist == -np.inf ] = 0.0
        ylist[ np.isnan(ylist)  ] = 0.0
        try:
            self.xlim_on_plot['min'] = min(xlist)
            self.xlim_on_plot['max'] = max(xlist)
            self.ylim_on_plot['min'] = min(ylist)
            self.ylim_on_plot['max'] = max(ylist)

        except ValueError:
            return

    # Adds nodes to the Tree View in Controls
    def addPlotToNode(self, dataset, parents):
        if dataset is None or parents is None:
            return

        # Get the pointer to the right Measurements and CorrectedDatasets
        for m in self.experiment.measurements:
            if m == parents['measurement']:
                measurement = m

        # Create the 'Other' CorrectedDatasets Node if it does not exist
        if 'Other' not in [m.title for m in measurement.corrected_datasets]:
            other = CorrectedDatasets(datasets=[dataset], title='Other')
            measurement.corrected_datasets.append(other)
        else:
            other = [m for m in measurement.corrected_datasets
                     if m.title == 'Other']
            if len(other) != 1:
                print("WARNING: More than 1 'Other' CorrectedDatsets...")
            other = other[0]
            other.datasets.append(dataset)

    # -------------------------------------------------------#
    # Dynamic

    # Gives the X,Y of the selected node and stores in selected_contents
    @property_depends_on('selected')
    def _get_selected_contents(self):
        if self.selected is None:
            return ''
        if isinstance(self.selected, Dataset):
            return self.selected.x, self.selected.y

    # Extracts Datasets models that are stored in the Experiment model
    @property_depends_on('experiment')
    def _get_datasets(self):
        datasets = list()
        for measurement in self.experiment.measurements:
            for corrected_dataset in measurement.corrected_datasets:
                for dataset in corrected_dataset.datasets:
                    # Strip +- inf 
                    y = dataset.y
                    y[ y ==  np.inf ] = 0.0
                    y[ y == -np.inf ] = 0.0

                    datasets.append(dataset)
        return datasets

    # Gets the selected Color Map, default == 'Set1'
    @property_depends_on('selected_cmap')
    def _get_selected_cmap_contents(self):
        if self.selected_cmap:
            return self.selected_cmap[0]
        return 'Set1'

    

    # Looks for change in Experiment and sets the correct limits
    @on_trait_change('experiment')
    def update_experiment(self):
        self.getLimitsForExperiment()
        self.xmin     = self.xlim_for_exp['min']
        self.xmin_min = self.xlim_for_exp['min']
        self.xmin_max = self.xlim_for_exp['max']

        self.xmax     = self.xlim_for_exp['max']
        self.xmax_min = self.xlim_for_exp['min']
        self.xmax_max = self.xlim_for_exp['max']

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
        myCMap = cm.get_cmap(name=self.controls.selected_cmap_contents)

        # Create a value for each line that corresponds to the color in the
        # ColorMap, ranged from 0->1
        cm_subsection = np.linspace(0.0, 1.0, num_lines)

        # Pull the colors from the ColorMap selected using the values created
        # in cm_subsection
        self._colors = [myCMap(x) for x in cm_subsection]

    # Use Controls X-range to select subset of the domain of the plot
    def _filter_xrange(self, xset, yset):
        xmin = self.controls.xmin
        xmax = self.controls.xmax

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
        self.controls.scale_factor = 1.0
        self.controls.shift_factor = 0.0

        # Pull the X, Y from the selected Dataset
        x = self.controls.selected.x
        y = self.controls.selected.y

        # Apply x-range filter
        x, y = self._filter_xrange(x, y)

        # Get the X, Y limits from all plots (selected + cached)
        self.controls.getLimitsOnPlot(x, y)

        # Set the limits
        axes.set_xlim(self.controls.xlim_on_plot['min'], self.controls.xlim_on_plot['max'])
        axes.set_ylim(self.controls.ylim_on_plot['min'], self.controls.ylim_on_plot['max'])

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
            axes.plot(x, y,label=dataset.title)

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
                self.controls = Controls(experiment=experiment)

    @on_trait_change('experiment_file.load_status')
    def update_status(self):
        self.status = self.experiment_file.load_status

    # Re-plot when we either select another Dataset or if we change the
    # ColorMap
    @on_trait_change('controls.selected,controls.selected_cmap')
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
    @on_trait_change(
        'controls.scale_factor,controls.shift_factor,'
        'controls.xmin,controls.xmax')
    def plot_dataset_modification(self):
        try:
            axes = self.get_axes()
            scale = self.controls.scale_factor
            shift = self.controls.shift_factor

            x = self.controls.selected.x
            y = scale * (self.controls.selected.y) + shift

            # Apply x-range filter
            x, y = self._filter_xrange(x, y)

            # Get the X, Y limits from all plots (selected + cached)
            self.controls.getLimitsOnPlot(x, y)

            # Set the limits
            axes.set_xlim(self.controls.xlim_on_plot['min'], self.controls.xlim_on_plot['max'])
            axes.set_ylim(self.controls.ylim_on_plot['min'], self.controls.ylim_on_plot['max'])

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

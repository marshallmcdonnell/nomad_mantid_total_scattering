#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import numpy as np

# Matplotlib
from matplotlib import cm

# Traits
from traits.api \
    import HasTraits, Instance, List, CFloat, Property, Any, \
    on_trait_change, property_depends_on

from traitsui.api \
    import RangeEditor, CheckListEditor, TextEditor, \
    View, HSplit, VSplit, VGroup, Item

# Local
from models \
    import Dataset, CorrectedDatasets, Experiment

from views \
    import ControlsView

# -----------------------------------------------------------#
# Node Controls Model


class NodeControls(HasTraits):
    pass

# -----------------------------------------------------------#
# Dataset Node Controls Model


class DatasetNodeControls(NodeControls):

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


    traits_view = View(
        VGroup(

            # Scale group
            HSplit(
                Item('scale_min', width=0.1,label='Min'),
                Item(
                    'scale_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='scale_min',
                        high_name='scale_max',
                        format='%4.2f',
                    ),
                    width=0.8,
                    show_label=False,
                ),
                Item('scale_max', width=0.1,label='Max'),
                show_border=True,
                label='Scale',
            ),

            # Shift group
            HSplit(
                Item('shift_min', width=0.1,label='Min'),
                Item(
                    'shift_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='shift_min',
                        high_name='shift_max',
                        format='%4.2f',
                    ),
                    width=0.8,
                    show_label=False,
                ),
                Item('shift_max', width=0.1,label='Max'),
                show_border=True,
                label='Shift',
            ),

            # X range
            VSplit(

                # Xmin
                HSplit(
                    Item('xmin_min',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          label='Min',
                    ),
                    Item('xmin',
                          editor=RangeEditor(
                              mode='slider',
                              low_name='xmin_min',
                              high_name='xmin_max',
                              format='%4.2f',
                          ),
                          width=0.8,
                          show_label=False,
                    ),
                    Item('xmin_max',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          label='Max',
                    ),
                    label='Xmin',
                ),

                # Xmax
                HSplit(
                    Item('xmax_min',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          label='Min',
                          ),
                    Item('xmax',
                          editor=RangeEditor(
                              mode='slider',
                              low_name='xmax_min',
                              high_name='xmax_max',
                              format='%4.2f',
                          ),
                          width=0.8,
                          show_label=False,
                          ),
                    Item('xmax_max',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          label='Max',
                          ),
                    label='Xmax',
                ),
                show_border=True,
                label='X-range',
            ),

            # Color map
            HSplit(
                Item('selected_cmap',
                      editor=CheckListEditor(name='cmap_list'),
                      show_label=False,
                ),
                show_border=True,
                label='ColorMap',
            ),
        ),
    )

    # Gets the selected Color Map, default == 'Set1'
    @property_depends_on('selected_cmap')
    def _get_selected_cmap_contents(self):
        if self.selected_cmap:
            return self.selected_cmap[0]
        return 'Set1'


# -----------------------------------------------------------#
# CorrectedDatasets Node Controls Model


class CorrectedDatasetsNodeControls(NodeControls):
    traits_view = View(width=0.1)

# -----------------------------------------------------------#
# Controls Model


class Controls(HasTraits):
    # View
    view = ControlsView

    # -------------------------------------------------------#
    # Traits

    # Passed in measurement
    experiment = Instance(Experiment, ())

    # Controls for selected node type
    node_controls = Instance(NodeControls)

    # The current list of datasets
    datasets = Property

    # The currently selected dataset
    selected = Any

    # The contents of the currently selected dataset
    selected_contents = Property

    # Node controls used for different types of TreeNodes
    # Cached plots we keep on plot
    cached_plots = List

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

        ylist[ylist == np.inf] = 0.0
        ylist[ylist == -np.inf] = 0.0
        ylist[np.isnan(ylist)] = 0.0
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
                    y[y == np.inf] = 0.0
                    y[y == -np.inf] = 0.0

                    datasets.append(dataset)
        return datasets

    # Looks for change in Experiment and sets the correct limits
    @on_trait_change('experiment')
    def update_experiment(self):
        self.getLimitsForExperiment()
        self.xmin = self.xlim_for_exp['min']
        self.xmin_min = self.xlim_for_exp['min']
        self.xmin_max = self.xlim_for_exp['max']

        self.xmax = self.xlim_for_exp['max']
        self.xmax_min = self.xlim_for_exp['min']
        self.xmax_max = self.xlim_for_exp['max']

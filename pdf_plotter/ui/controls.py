#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import numpy as np

# Matplotlib
from matplotlib import cm

# Traits
from traits.api \
    import HasTraits, Instance, List, CFloat, Property, \
    Any, Button, Event, Bool, Str,\
    on_trait_change, property_depends_on, cached_property

from traitsui.api \
    import RangeEditor, CheckListEditor, TextEditor, InstanceEditor, \
    View, HSplit, VSplit, HGroup, VGroup, Item, Action

# Local
import ui.models as models 
import ui.views  as views
import ui.controllers  as controllers

# -----------------------------------------------------------#
# Generic Node Models

class NodeButtons(HasTraits):
    button_event = Bool(False)
    button_name  = Str

class NodeControls(HasTraits):
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


    # Use X-range to select subset of the domain of the datasets
    def filter_xrange(self, xset, yset):
        xmin = self.xmin
        xmax = self.xmax

        xout = list()
        yout = list()

        for x, y in zip(xset, yset):
            if xmin <= x and x <= xmax:
                xout.append(x)
                yout.append(y)

        return xout, yout

    # Gets the selected Color Map, default == 'Set1'
    @property_depends_on('selected_cmap')
    def _get_selected_cmap_contents(self):
        if self.selected_cmap:
            return self.selected_cmap[0]
        return 'Set1'

    @on_trait_change('xmin')
    def update_xmin_xmax(self):
        if self.xmin < self.xmin_min:
            self.xmin_min = self.xmin
        if self.xmin > self.xmin_max:
            self.xmin_max = self.xmin

        if self.xmax < self.xmax_min:
            self.xmax_min = self.xmax
        if self.xmax > self.xmax_max:
            self.xmax_max = self.xmax


# -----------------------------------------------------------#
# Dataset Node Models

class DatasetNodeButtons(NodeButtons):
    cache_button       = Button
    clear_cache_button = Button
    
    traits_view = View(
            HGroup( 
                Item('cache_button',
                     label="Cache Plot",
                     show_label=False,
                ),
                Item('clear_cache_button',
                     label="Clear Cache",
                     show_label=False,
                ),
            ),
        handler=controllers.DatasetNodeButtonHandler(),
    )

class DatasetNodeControls(NodeControls):

    # Scale controls
    scale_min = CFloat(0.5)
    scale_max = CFloat(1.5)
    scale_factor = CFloat(1.0)

    # Scale controls
    shift_min = CFloat(-5.0)
    shift_max = CFloat(5.0)
    shift_factor = CFloat(0.0)

    traits_view = View(
        VGroup(

            # Scale group
            HSplit(
                Item('scale_min', width=0.1, label='Min'),
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
                Item('scale_max', width=0.1, label='Max'),
                show_border=True,
                label='Scale',
            ),

            # Shift group
            HSplit(
                Item('shift_min', width=0.1, label='Min'),
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
                Item('shift_max', width=0.1, label='Max'),
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


# -----------------------------------------------------------#
# CorrectedDatasets Node Models

class CorrectedDatasetsNodeButtons(NodeButtons):
    cache_button       = Instance(Button("Cache Plots"))

class CorrectedDatasetsNodeControls(NodeControls):
    traits_view = View(
        HSplit(
            Item('selected_cmap',
                 editor=CheckListEditor(name='cmap_list'),
                 show_label=False,
                 ),
            show_border=True,
            label='ColorMap',
        ),
    )

# -----------------------------------------------------------#
# Controls Model


class Controls(HasTraits):
    # View
    view = views.ControlsView

    # -------------------------------------------------------#
    # Traits

    # Passed in measurement
    experiment = Instance(models.Experiment, ())

    # Controls for selected node type
    node_controls = Instance(NodeControls)

    # Buttons for the selected node type
    node_buttons = Instance(NodeButtons)

    # The currently selected dataset
    selected = Any

    # The contents of the currently selected dataset
    selected_contents = Property

    # Node controls used for different types of TreeNodes
    # Cached plots we keep on plot
    cached_plots = List

    # Cached properties for the loaded experiment
    datasets = Property(depends_on='experiment')
    xlist = Property(depends_on='datasets')
    ylist = Property(depends_on='datasets')

    exp_xmin = Property(depends_on='xlist')
    exp_xmax = Property(depends_on='xlist')

    # -------------------------------------------------------#
    # Utilities

    # Adds nodes to the Tree View in Controls
    def add_plot_to_node(self, dataset, parents):
        if dataset is None or parents is None:
            return

        # Get the pointer to the right Measurements and CorrectedDatasets
        for m in self.experiment.measurements:
            if m == parents['measurement']:
                measurement = m

        # Create the 'Other' CorrectedDatasets Node if it does not exist
        if 'Other' not in [m.title for m in measurement.corrected_datasets]:
            other = models.CorrectedDatasets(datasets=[dataset], title='Other')
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
        if isinstance(self.selected, models.Dataset):
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

    # Get a list of all x values for all datasets
    @cached_property
    def _get_xlist(self):
        xlist = []
        for d in self.datasets:
            xlist = np.append(xlist, d.x, axis=None)
        return xlist

    @cached_property
    def _get_exp_xmin(self):
        return min(self.xlist)

    @cached_property
    def _get_exp_xmax(self):
        return max(self.xlist)

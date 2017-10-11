from __future__ import (absolute_import, division, print_function)

from traitsui.api \
    import Handler

import models 
import controls

# -----------------------------------------------------------#
# Controllers

class NodeButtonHandler(Handler):
    def trigger_button_event(self, info):
        info.object.button_event = True
        info.object.button_event = False

class DatasetNodeButtonHandler(NodeButtonHandler):
    def object_cache_button_changed(self, info):
        info.object.button_name  = 'cache_plot'
        self.trigger_button_event(info)

    def object_clear_cache_button_changed(self, info):
        info.object.button_name  = 'clear_cache'
        self.trigger_button_event(info)

class CorrectedDatasetsNodeButtonHandler(NodeButtonHandler):
    def object_cache_button_changed(self, info):
        info.object.button_name  = 'cache_plots'
        self.trigger_button_event(info)

    def object_clear_cache_button_changed(self, info):
        info.object.button_name  = 'clear_cache'
        self.trigger_button_event(info)


class ControlPanelHandler(Handler):
    # -----------------------------------------------------------#
    def get_parents(self, info, node):
        # Grab selected Dataset
        selected = info.object.controls.selected

        # Get the TreeEditor for the Experiment
        controls_editors = info.ui.get_editors("controls")
        experiment_editors = list()
        for editor in controls_editors:
            experiment_editors.extend(editor._ui.get_editors("experiment"))
        experiment_editor = experiment_editors[0]  # just grab first

        # Get the parents
        corrected_dataset = experiment_editor.get_parent(selected)
        measurement = experiment_editor.get_parent(corrected_dataset)
        experiment = experiment_editor.get_parent(measurement)

        parents = {'corrected_dataset': corrected_dataset,
                   'measurement': measurement,
                   'experiment': experiment}

        return parents

    def object_button_pressed_changed(self, info):
        if info.initialized:
            if info.object.controls.node_buttons.button_event:
                if info.object.controls.node_buttons.button_name == 'cache_plot':
                    self.cache_plot(info)
                if info.object.controls.node_buttons.button_name == 'clear_cache':
                    self.clear_cache(info)
                if info.object.controls.node_buttons.button_name == 'cache_plots':
                    self.cache_plots(info)

            # Reset the button status
            #info.object.controls.node_buttons.button_event = False

    def cache_plot(self, info):
        selected = info.object.controls.selected

        # Get info for selected Dataset (=a) and create new Dataset (=b)
        a = selected
        shift = info.object.controls.node_controls.shift_factor
        scale = info.object.controls.node_controls.scale_factor
        b = models.Dataset(x=a.x, y=scale * a.y + shift, title=a.title)

        # Apply x-range filter
        b.x, b.y = info.object.controls.node_controls.filter_xrange(b.x, b.y)

        # If we have modified Dataset 'a', change title of 'b' for
        # differences in...
        tmp_title = str(b.title)

        try:
            # Shift
            if shift != 0.0:
                b.title += " shift: {0:>5.2f}".format(shift)

            # Scale
            if scale != 1.0:
                b.title += " scale: {0:>5.2f}".format(scale)

            # Xmin
            if min(b.x) != min(a.x):
                b.title += " xmin: {0:.2f}".format(min(b.x))

            # Xmax
            if max(b.x) != max(a.x):
                b.title += " xmax: {0:.2f}".format(max(b.x))

            # Check if title changed.
            # If so, add as a different Node in 'Other' Measurement
            if tmp_title != b.title:
                parents = self.get_parents(info, b)
                info.object.controls.add_plot_to_node(dataset=b,
                                                      parents=parents)

            # Add 'b' to cached plots
            info.object.controls.cached_plots.append(b)

            # Add to plot and refresh
            axes = info.object.get_axes()
            axes.plot(b.x, b.y, label=b.title)
            info.object._setupColorMap(info.object.controls.cached_plots)
            info.object.plot_cached()
            info.object.plot_dataset_modification()

        except ValueError:
            pass


    def cache_plots(self, info):
        datasets = info.object.controls.selected.datasets
        axes = info.object.get_axes()
        for dataset in datasets:
            axes.plot(dataset.x, dataset.y, label=dataset.title)
            info.object.controls.cached_plots.append(dataset)    

        info.object._setupColorMap(info.object.controls.cached_plots)
        info.object.plot_cached()
        info.object.plot_dataset_modification()


    def clear_cache(self, info):
        info.object.controls.cached_plots = []
        axes = info.object.get_axes()
        axes.cla()
        info.object.plot_dataset_modification()

    def object_selected_changed(self, info):
        if not info.initialized:
            return

        xmin = info.object.controls.exp_xmin
        xmax = info.object.controls.exp_xmax
        selected_cmap = info.object.controls.node_controls.selected_cmap
        if isinstance(info.object.selected, models.Dataset):
            info.object.controls.node_controls = controls.DatasetNodeControls(
                xmin=xmin,
                xmin_min=xmin,
                xmin_max=xmax,
                xmax=xmax,
                xmax_min=xmin,
                xmax_max=xmax,
                selected_cmap=selected_cmap,
            )

            info.object.controls.node_buttons = controls.DatasetNodeButtons()

        elif isinstance(info.object.selected, models.CorrectedDatasets):
            info.object.controls.node_controls = controls.CorrectedDatasetsNodeControls(
                xmin=xmin,
                xmax=xmax,
                selected_cmap=selected_cmap,
            )

            info.object.controls.node_buttons = controls.CorrectedDatasetsNodeButtons()


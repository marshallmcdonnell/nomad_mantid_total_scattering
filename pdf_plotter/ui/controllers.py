from __future__ import (absolute_import, division, print_function)

from traitsui.api \
    import Handler

from models \
    import Dataset, CorrectedDatasets

from controls \
    import DatasetNodeControls, CorrectedDatasetsNodeControls

# -----------------------------------------------------------#
# Controllers

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


    def cache_plot(self, info):
        selected = info.object.controls.selected
        if isinstance(selected, Dataset):

            # Get info for selected Dataset (=a) and create new Dataset (=b)
            a = selected
            shift = info.object.controls.node_controls.shift_factor
            scale = info.object.controls.node_controls.scale_factor
            b = Dataset(x=a.x, y=scale * a.y + shift, title=a.title)

            # Apply x-range filter
            b.x, b.y = info.object._filter_xrange(b.x, b.y)

            # If we have modified Dataset 'a', change title of 'b' for
            # differences in...
            tmp_title = str(b.title)

            try:
                # Shift
                if shift != 0.0:
                    b.title += " shift: {0:.2}".format(shift)

                # Scale
                if scale != 0.0:
                    b.title += " scale: {0:.2}".format(scale)

                # Xmin
                if min(b.x) != min(a.x):
                    b.title += " xmin: {0:.2}".format(min(b.x))

                # Xmax
                if max(b.x) != max(a.x):
                    b.title += " xmax: {0:.2}".format(max(b.x))

                # Check if title changed.
                # If so, add as a different Node in 'Other' Measurement
                if tmp_title != b.title:
                    parents = self.get_parents(info, b)
                    info.object.controls.addPlotToNode(dataset=b,
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
                return

    def clear_cache(self, info):
        info.object.controls.cached_plots = []
        axes = info.object.get_axes()
        axes.cla()
        info.object.plot_dataset_modification()


    
    def object_selected_changed(self, info):
        if not info.initialized:
            return

        if isinstance(info.object.selected, Dataset):
            info.object.controls.node_controls = DatasetNodeControls() 

        elif isinstance(info.object.selected, CorrectedDatasets):
            info.object.controls.node_controls = CorrectedDatasetsNodeControls() 


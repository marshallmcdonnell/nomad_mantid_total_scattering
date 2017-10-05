from __future__ import (absolute_import, division, print_function)
from traitsui.api \
    import Action, Handler

from models \
    import Dataset


# -----------------------------------------------------------#
# Actions

CachePlotAction = Action(name="Cache Plot",
                         action="cache_plot")

ClearCacheAction = Action(name="Clear Cache",
                          action="clear_cache")


# -----------------------------------------------------------#
# Controllers
class ControlPanelHandler(Handler):

    def cache_plot(self, info):
        selected = info.object.controls.selected
        if isinstance(selected, Dataset):

            # Get info for selected Dataset (=a) and create new Dataset (=b)
            a = selected
            shift = info.object.controls.shift_factor
            scale = info.object.controls.scale_factor
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
                    info.object.controls.addPlotToNode(b)

                # Add 'b' to cached plots
                info.object.controls.cached_plots.append(b)

                # Add to plot and refresh
                axes = info.object._get_axes()
                axes.plot(b.x, b.y, label=b.title)
                info.object.plot_modification()

            except ValueError:
                return

    def clear_cache(self, info):
        info.object.controls.cached_plots = []
        axes = info.object._get_axes()
        axes.cla()
        info.object.plot_modification()

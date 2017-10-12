#!/usr/bin/env python

from traits.api \
    import HasTraits, Bool, Str, CFloat, List, Any, Property, \
    property_depends_on, on_trait_change

from traitsui.api \
    import Handler

from matplotlib import cm

# -----------------------------------------------------------#
# Generic Node Buttons


class NodeButtonHandler(Handler):
    def trigger_button_event(self, info):
        info.object.button_event = True
        info.object.button_event = False


class NodeButtons(HasTraits):
    button_event = Bool(False)
    button_name = Str

# -----------------------------------------------------------#
# Generic Node Controls


class NodeControls(HasTraits):
    # Node selected in Controls
    selected = Any

    # Freeze limits
    freeze_xlims = Bool(False)
    freeze_ylims = Bool(False)

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
    def filter_xrange(self, xset, yset, dataset):
        print("filter_xrange 1 ")
        xmin = dataset.xmin_filter
        xmax = dataset.xmax_filter
        print(xmin, xmax)
        print("filter_xrange 2 ")

        xout = list()
        yout = list()

        for x, y in zip(xset, yset):
            if xmin <= x and x <= xmax:
                xout.append(x)
                yout.append(y)

        print("filter_xrange 3 ")
        return xout, yout

    # Gets the selected Color Map, default == 'Set1'
    @property_depends_on('selected_cmap')
    def _get_selected_cmap_contents(self):
        if self.selected_cmap:
            return self.selected_cmap[0]
        return 'Set1'


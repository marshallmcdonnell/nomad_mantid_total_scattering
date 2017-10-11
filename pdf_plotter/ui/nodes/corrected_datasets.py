#!/usr/bin/env python

from traits.api \
    import Button

from traitsui.api \
    import View, HGroup, HSplit, Item, CheckListEditor

from ui.nodes.base_node \
    import NodeButtonHandler, NodeButtons, NodeControls

# -----------------------------------------------------------#
# CorrectedDatasets Node Buttons


class CorrectedDatasetsNodeButtonHandler(NodeButtonHandler):
    def object_cache_button_changed(self, info):
        info.object.button_name = 'cache_plots'
        self.trigger_button_event(info)

    def object_clear_cache_button_changed(self, info):
        info.object.button_name = 'clear_cache'
        self.trigger_button_event(info)


class CorrectedDatasetsNodeButtons(NodeButtons):
    cache_button = Button
    clear_cache_button = Button

    traits_view = View(
        HGroup(
            Item('cache_button',
                 label="Cache Plots",
                 show_label=False,
                 ),
            Item('clear_cache_button',
                 label="Clear Cache",
                 show_label=False,
                 ),
        ),
        handler=CorrectedDatasetsNodeButtonHandler(),
    )


# -----------------------------------------------------------#
# CorrectedDatasets Node Controls

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


from traitsui.api \
    import TableEditor, RangeEditor, CheckListEditor, \
    InstanceEditor, TextEditor, spring, \
    View, HSplit, VSplit, HGroup, VGroup, Group, UItem, Item

from traitsui.table_column \
    import ObjectColumn

from mpl_utilities \
    import MPLFigureEditor

from editors import ExperimentTreeEditor

from controllers import CachePlotAction, ClearCacheAction


# -----------------------------------------------------------#
# Simpl Table Editor for Views

table_editor = TableEditor(
    columns=[ObjectColumn(name='title', editable=False, width=0.3)],
    selected='selected',
    auto_size=False,
)

# -----------------------------------------------------------#
# Views

MeasurementView = View(
    Item(
        'datasets',
        show_label=False,
        editor=table_editor),
    resizable=True,
)
ExperimentView = View(
    Item(
        'measurements',
        show_label=False,
        editor=table_editor),
    resizable=True,
)

SofqPlotView = View(
    Item(
        'figure',
        editor=MPLFigureEditor(),
        show_label=False,
        resizable=True,
        springy=True))

ControlsView = View(
    VSplit(
        UItem(
            name='experiment',
            editor=ExperimentTreeEditor,
            resizable=True,
            show_label=False,
            width=0.9),
        VGroup(
            HSplit(
                UItem('scale_min', width=0.1),
                UItem(
                    'scale_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='scale_min',
                        high_name='scale_max',
                        format='%4.2f',
                    ),
                    width=0.8,
                ),
                UItem('scale_max', width=0.1),
                show_border=True,
                label='Scale',
            ),
            HSplit(
                UItem('shift_min', width=0.1),
                UItem(
                    'shift_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='shift_min',
                        high_name='shift_max',
                        format='%4.2f',
                    ),
                    width=0.8,
                ),
                UItem('shift_max', width=0.1),
                show_border=True,
                label='Shift',
            ),
            HSplit(
                UItem('selected_cmap',
                      editor=CheckListEditor(name='cmap_list')
                ),
                show_border=True,
                label='ColorMap',
            ),
        ),
    ))

ControlPanelView = View(
    HSplit(
        UItem('sofq_plot', width=0.7, style='custom', editor=InstanceEditor()),
        UItem('controls', width=0.3, style='custom', editor=InstanceEditor()),
    ),
    buttons=[CachePlotAction, ClearCacheAction],
    resizable=True,
)

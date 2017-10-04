
from traitsui.api \
    import TableEditor, RangeEditor, CheckListEditor, \
    InstanceEditor, TextEditor, \
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
    VGroup(
        UItem(
            name='experiment',
            editor=ExperimentTreeEditor,
            resizable=True,
            show_label=False,),
        VGroup(
            HGroup(
                UItem('scale_min'),
                UItem(
                    'scale_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='scale_min',
                        high_name='scale_max',
                        format='%4.2f')),
                UItem('scale_max'),
                show_border=True,
                label='Scale',
            ),
            HGroup(
                UItem('shift_min'),
                UItem(
                    'shift_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='shift_min',
                        high_name='shift_max',
                        format='%4.2f')),
                UItem('shift_max'),
                show_border=True,
                label='Shift',
            ),
            UItem(
                'selected_cmap',
                editor=CheckListEditor(
                    name='cmap_list')),
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

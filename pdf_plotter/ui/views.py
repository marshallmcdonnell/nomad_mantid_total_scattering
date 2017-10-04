
from traitsui.api \
    import TableEditor, RangeEditor, CheckListEditor, \
    InstanceEditor, TextEditor, \
    View, HGroup, VGroup, Group, UItem, Item

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
            show_label=False),
        Group(
            HGroup(
                UItem('scale_min'),
                UItem(
                    'scale_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='scale_min',
                        high_name='scale_max')),
                UItem('scale_max'),
                show_border=True,
            ),
            HGroup(
                UItem('shift_min'),
                UItem(
                    'shift_factor',
                    editor=RangeEditor(
                        mode='slider',
                        low_name='shift_min',
                        high_name='shift_max')),
                UItem('shift_max'),
            ),
            UItem(
                'selected_cmap',
                editor=CheckListEditor(
                    name='cmap_list')),
        ),
    ))

ControlPanelView = View(
    HGroup(
        UItem('sofq_plot', width=500, style='custom', editor=InstanceEditor()),
        UItem('controls', width=200, style='custom', editor=InstanceEditor()),
    ),
    buttons=[CachePlotAction, ClearCacheAction],
    resizable=True,
)

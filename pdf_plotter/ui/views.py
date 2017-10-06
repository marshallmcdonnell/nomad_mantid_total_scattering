
from traitsui.api \
    import TableEditor, RangeEditor, CheckListEditor, \
    InstanceEditor, TextEditor, \
    View, HSplit, VSplit, VGroup, UItem, Item, StatusItem

from traitsui.table_column \
    import ObjectColumn

from mpl_utilities \
    import MPLFigureEditor

from editors import ExperimentTreeEditor

from controllers \
    import CachePlotAction, ClearCacheAction


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
    Item('datasets',
         show_label=False,
         editor=table_editor
         ),
    resizable=True,
)

ExperimentView = View(
    Item('measurements',
         show_label=False,
         editor=table_editor
         ),
    resizable=True,
)

ExperimentFileInputView = View(
    Item('load_button',
         show_label=False,
         ),
)

SofqPlotView = View(
    Item('figure',
         editor=MPLFigureEditor(),
         show_label=False,
         resizable=True,
         springy=True
         ),
)

ControlsView = View(
    VSplit(
        # Experiment Tree
        UItem(
            name='experiment',
            editor=ExperimentTreeEditor,
            resizable=True,
            show_label=False,
            width=0.9
        ),

        # Tools
        VGroup(

            # Scale group
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

            # Shift group
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

            # X range
            VSplit(

                # Xmin
                HSplit(
                    UItem('xmin_min',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          ),
                    UItem('xmin',
                          editor=RangeEditor(
                              mode='slider',
                              low_name='xmin_min',
                              high_name='xmin_max',
                              format='%4.2f',
                          ),
                          width=0.8,
                          ),
                    UItem('xmin_max',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          ),
                    label='Xmin',
                ),

                # Xmax
                HSplit(
                    UItem('xmax_min',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          ),
                    UItem('xmax',
                          editor=RangeEditor(
                              mode='slider',
                              low_name='xmax_min',
                              high_name='xmax_max',
                              format='%4.2f',
                          ),
                          width=0.8,
                          ),
                    UItem('xmax_max',
                          width=0.1,
                          editor=TextEditor(auto_set=False,),
                          ),
                    label='Xmax',
                ),
                show_border=True,
                label='X-range',
            ),

            # Color map
            HSplit(
                UItem('selected_cmap',
                      editor=CheckListEditor(name='cmap_list')
                      ),
                show_border=True,
                label='ColorMap',
            ),
        ),
    ),
)

ControlPanelView = View(
    HSplit(
        UItem('sofq_plot', width=0.7, style='custom', editor=InstanceEditor()),
        VSplit(
            UItem('experiment_file',
                  height=0.1,
                  style='custom',
                  editor=InstanceEditor()
                  ),
            UItem('controls',
                  height=0.9,
                  style='custom',
                  editor=InstanceEditor()
                  ),
        ),
    ),
    buttons=[CachePlotAction, ClearCacheAction],
    resizable=True,
    statusbar=[StatusItem(name='load_status')]
)

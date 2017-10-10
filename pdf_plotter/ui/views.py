
from traitsui.api \
    import TableEditor, InstanceEditor, \
    Action, View, HSplit, VSplit, UItem, Item, StatusItem

from traitsui.table_column \
    import ObjectColumn

from mpl_utilities \
    import MPLFigureEditor

import editors

# -----------------------------------------------------------#
# Simple Table Editor for Views

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
            editor=editors.ExperimentTreeEditor,
            resizable=True,
            show_label=False,
            height=0.7,
        ),
        UItem('node_controls',
              editor=InstanceEditor(),
              style='custom',
              resizable=True,
              height=0.2,
              ),
        UItem('node_buttons',
              editor=InstanceEditor(),
              style='custom',
              resizable=True,
              height=0.1,
              show_label=False,
              ),
    ),
)

ControlPanelView = View(
    HSplit(
        UItem('sofq_plot', width=0.8, style='custom', editor=InstanceEditor()),
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
    resizable=True,
    statusbar=[StatusItem(name='load_status')]
)

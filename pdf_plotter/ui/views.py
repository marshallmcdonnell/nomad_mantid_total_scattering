
from traitsui.api \
    import TableEditor, RangeEditor, View, HGroup, VGroup, Group, Item

from traitsui.table_column \
    import ObjectColumn

from mpl_utilities \
    import MPLFigureEditor

from editors import ExperimentTreeEditor

from controllers import PrintHelpAction


#-----------------------------------------------------------#
# Simpl Table Editor for Views

table_editor= TableEditor(
                           columns= [ObjectColumn(name='title', editable=False, width=0.3)],
                           selected='selected',
                           auto_size=False,
                         )

#-----------------------------------------------------------#
# Views
 
MeasurementView  = View(Item('datasets', show_label=False, editor=table_editor),resizable=True, )
ExperimentView   = View(Item('measurements', show_label=False, editor=table_editor),resizable=True, )
ControlPanelView = View(
                        HGroup(
                            Item('figure', editor=MPLFigureEditor(), show_label=False),
                            VGroup(
                                Item(name='experiment',
                                     editor=ExperimentTreeEditor,
                                     resizable=True,
                                     show_label=False),
                                Group(
                                    Item('scale', editor=RangeEditor(mode='slider')),  
                                    Item('shift', editor=RangeEditor(mode='xslider')),
                                ),
                            ),
                        ),
                buttons = [ PrintHelpAction ],
                resizable=True,
)


#!/usr/bin/env python

from traits.api \
    import HasTraits, Array, Str, Instance, List, Float,\
           DelegatesTo, Property, Any, \
           on_trait_change, property_depends_on

from traitsui.api \
    import Group, View, Item, TableEditor, RangeEditor, \
           ListEditor, EnumEditor, HGroup, VGroup

from traitsui.table_column \
    import ObjectColumn

from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np

from mpl_interaction import PanAndZoom

#-----------------------------------------------------------#
# Matplotlib w/ Qt4 classes for TraitsUI Editor

class _MPLFigureEditor(Editor):
    
    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        mpl_canvas = FigureCanvas(self.value)
        return mpl_canvas

class MPLFigureEditor(BasicEditorFactory):

   klass = _MPLFigureEditor

#-----------------------------------------------------------#
# Models

class Dataset(HasTraits):
    x = Array
    y = Array
    title = Str


class Measurement(HasTraits):
    datasets = List(Dataset)



table_editor= TableEditor(
                           columns= [ObjectColumn(name='title', editable=False, width=0.3)], 
                           selected='selected',
                           auto_size=False,
                         )

measurement_view = View( 
                        Item('datasets', show_label=False, editor=table_editor),
                        resizable=True,
                       )



class ControlPanel(HasTraits):
    # Passed in measurement
    measurement = Instance(Measurement, ())

    # Figure to display selected dataset
    figure = Instance(Figure, ())

    # The current list of datasets
    datasets  = Property

    # The currently selected dataset
    selected = Any
    
    # The contents of the currently selected dataset
    selected_contents = Property

    # Scale and shift controls
    scale = Float(1.0)
    shift = Float(0.0)

    view = View(
             HGroup(
                    Item('figure', editor=MPLFigureEditor(), show_label=False),
                    VGroup(
                           Item('datasets', editor=table_editor),
                           Group(
                            Item('scale', editor=RangeEditor(mode='slider')),  
                            Item('shift', editor=RangeEditor(mode='xslider')),
                           ),
                          ),
                ),
                resizable=True)

    def _get_datasets(self):
        return [dataset for dataset in self.measurement.datasets]
            

    @property_depends_on('selected')
    def _get_selected_contents(self):
        if self.selected is None:
            return ''

        return self.selected.x, self.selected.y

    @on_trait_change('selected,scale,shift')
    def plot(self):
        scale = self.scale
        shift = self.shift

        x = self.selected.x
        y = scale*(self.selected.y) + shift 
   
        self.figure.pan_zoom = PanAndZoom(self.figure)
        axes = self.figure.add_subplot(111)
        
        if not axes.lines:
            axes.plot(x, y)
        else:
            l = axes.lines[0]
            l.set_xdata(x)
            l.set_ydata(y)

        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()


   

if __name__ == "__main__":

    x = np.linspace(0,4*np.pi, 200)
    y = np.sin(x)
    d1 = Dataset(x=x,y=y,title='sin(x)') 

    y = np.sin(x) * np.sin(x)
    d2 = Dataset(x=x,y=y,title='sin(x) * sin(x)') 

    m1 = Measurement(datasets=[d1, d2])

    m1.configure_traits(view=measurement_view)

    cp = ControlPanel(measurement=m1)
    cp.configure_traits()

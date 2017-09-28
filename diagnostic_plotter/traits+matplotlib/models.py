#!/usr/bin/env python

from traits.api \
    import HasTraits, Array, Str, Instance, List, Float,\
           DelegatesTo, Property, Any, \
           on_trait_change, property_depends_on

from traitsui.api \
    import Group, View, Item, TableEditor, RangeEditor, \
           TreeEditor, TreeNode, HGroup, VGroup

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
    title = Str

class Experiment(HasTraits):
    measurements = List(Measurement) 
    title = Str

table_editor= TableEditor(
                           columns= [ObjectColumn(name='title', editable=False, width=0.3)], 
                           selected='selected',
                           auto_size=False,
                         )

measurement_view = View(Item('datasets', show_label=False, editor=table_editor),resizable=True, )
experiment_view = View(Item('measurements', show_label=False, editor=table_editor),resizable=True, )

tree_editor = TreeEditor(
                  nodes = [
                            TreeNode( node_for  = [ Experiment ],
                                      auto_open = True,
                                      children  = '',
                                      label     = 'title',
                                      view      = View( Group('title', orientation='vertical', show_left=True))),
                            TreeNode( node_for  = [ Experiment ],
                                      auto_open = True,
                                      children  = 'measurements',
                                      label     = '=Measurements',
                                      view      = View(),
                                      add       = [ Measurement ] ),
                            TreeNode( node_for  = [ Measurement ],
                                      auto_open = True,
                                      children  = 'datasets',
                                      label     = 'title',
                                      view      = View( Group('title', orientation='vertical', show_left=True)),
                                      add       = [ Dataset ] ),
                            TreeNode( node_for  = [ Dataset ],
                                      auto_open = True,
                                      label     = 'title',
                                      view      = View()),
                          ],
                 selected='selected'
)


class ControlPanel(HasTraits):
    # Passed in measurement
    experiment = Instance(Experiment, ())

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
                           Item(name='experiment',editor=tree_editor,resizable=True),
                           Group(
                            Item('scale', editor=RangeEditor(mode='slider')),  
                            Item('shift', editor=RangeEditor(mode='xslider')),
                           ),
                          ),
                ),
                resizable=True)

    @property_depends_on('experiment')
    def _get_datasets(self):
        datasets = list()
        for measurement in self.experiment.measurements:
            for dataset in measurement.datasets:
                datasets.append(dataset)
        return datasets
            

    @property_depends_on('selected')
    def _get_selected_contents(self):
        if self.selected is None:
            return ''

        return self.selected.x, self.selected.y

    @on_trait_change('selected')
    def plot_selection(self):
        try:
            x = self.selected.x
            y = self.selected.y 

            self.figure.pan_zoom = PanAndZoom(self.figure)

            axes = self.figure.add_subplot(111)
            axes.set_xlim(min(x),max(x))
            axes.set_ylim(min(y),max(y))

            self.plot(axes, x, y)
    
        except AttributeError:
            pass

    @on_trait_change('scale,shift')
    def plot_modification(self):
        try:
            scale = self.scale
            shift = self.shift

            x = self.selected.x
            y = scale*(self.selected.y) + shift 

            axes = self.figure.add_subplot(111)

            self.plot(axes, x, y)
        except AttributeError:
            pass
 
    def plot(self, axes, x, y):

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

    # Make two datsets w/ titles
    x = np.linspace(0,4*np.pi, 200)
    y = np.sin(x)
    d1 = Dataset(x=x,y=y,title='sin(x)') 

    y = np.sin(x) * np.sin(x)
    d2 = Dataset(x=x,y=y,title='sin(x) * sin(x)') 

    # Use the datasets to make a Measurement
    m1 = Measurement(datasets=[d1, d2], title='sine functions')
    #m1.configure_traits(view=measurement_view)

    # Now make a second measurement
    x = np.linspace(-20,20,200)
    m2 = Measurement( datasets = [ Dataset(x=x, y=x*x, title='x^2'),
                                   Dataset(x=x, y=x*x*x, title='x^3') ],
                      title='polynomials' )

    # Create a Experiment from these two measurements
    e1 = Experiment(measurements=[m1,m2],title='Test Functions')

    # Use the ControlPanel to View the Measurement
    cp = ControlPanel(experiment=e1)
    cp.configure_traits()

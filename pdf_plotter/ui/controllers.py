from __future__ import (absolute_import, division, print_function)
from traitsui.api \
    import Action, Handler

from models \
    import Dataset


#-----------------------------------------------------------#
# Actions

CachePlotAction = Action(name="Cache Plot",
                         action="cache_plot")

ClearCacheAction = Action(name="Clear Cache",
                          action="clear_cache")


#-----------------------------------------------------------#
# Controllers
class ControlPanelHandler(Handler):

    def cache_plot(self, info):
        selected_type = type(info.object.selected)
        if selected_type is Dataset:

            # Get info for selected Dataset (=a) and create new Dataset (=b)
            a = info.object.selected
            shift = info.object.shift
            scale = info.object.scale
            b = Dataset(x=a.x,y=scale*a.y+shift,title=a.title)

            # If we have modified Dataset 'a', change title of 'b' 
            # and add to 'Other' Measurement in Experiment Tree
            if shift != 0.0 or scale != 1.0:
                b.title += " shift: {0:.2} scale: {1:.2}".format(shift, scale)
                info.object.addPlotToNode(b)

            # Add 'b' to cached plots
            info.object.cached_plots.append(b)

            # Add to plot and refresh
            info.object.axes.plot(b.x, b.y, label=b.title)
            info.object.plot_modification()

    def clear_cache(self,info):
        info.object.cached_plots = []
        info.object.axes.cla()
        info.object.plot_modification()
            
        
        

from traitsui.api \
    import Action, Handler

from models \
    import Dataset


#-----------------------------------------------------------#
# Actions

PrintHelpAction = Action(name="Cache Plot",
                         action="cache_plot")


#-----------------------------------------------------------#
# Controllers
class ControlPanelHandler(Handler):

    def cache_plot(self, info):
        print "Cached_plots:", len(info.object.cached_plots)
        mtype = type(info.object.selected)
        print mtype, mtype is Dataset
        if type(info.object.selected) is Dataset:
            info.object.cached_plots.append(info.object.selected)
        
        


import threading
import numpy as np
from nexusformat import nexus

from models \
    import Dataset


# -----------------------------------------------------------#
# Thread to handle loading in Experiment files

class NexusFileThread(threading.Thread):
    def __init__(self, f):
        self.f = f
        self.nxresult = None
        threading.Thread.__init__(self)

    # Multithreaded extraction of Datasets from Nexus file
    def extract_datasets_nexus(self):
        nx = self.nxresult
        a = [nx[wksp] for wksp in nx]
        print(a)
        wksps = [nx[wksp]
                 for wksp in nx if str(wksp.title).startswith("mantid_workspace")]

        t_list = list()
        for i, wksp in enumerate(wksps):
            t = DatasetThread(wksp)
            t.datasets = self.datasets
            t.start()
            t_list.append(t)

        for t in t_list:
            t.join()

    # Main thread opens and extracts Nexus and then launchs threads to extract
    # Datasets
    def run(self):
        self.update_status("Loading...")
        with nexus.NXFile(self.f, 'r') as f:
            self.nxresult = f.readfile()
        self.update_status("Done Loading!")
        self.update_status('Extracting Data...')
        self.extract_datasets_nexus()
        self.update_status("Done Extracting!")
        return

# -----------------------------------------------------------#
# Thread to handle extracting Datasets


class DatasetThread(threading.Thread):
    def __init__(self, wksp):
        self.wksp = wksp
        threading.Thread.__init__(self)

    def sort_lists(self, sorter=None, sortee=None):
        if sorter is None or sortee is None:
            return
        sorter_result = [x for x, y in sorted(
            zip(sorter, sortee), key=lambda pair: pair[0])]
        sortee_result = [y for x, y in sorted(
            zip(sorter, sortee), key=lambda pair: pair[0])]
        return sorter_result, sortee_result

    def getTag(self, title):
        if title.startswith('sample'):
            return 'Sample'
        elif title.startswith('vanadium'):
            return 'Vanadium'
        elif title.startswith('container_background'):
            return 'Container Background'
        elif title.startswith('container'):
            return 'Container'
        elif title.startswith('correction'):
            return 'Correction'
        else:
            return 'Other'

    def run(self):
        wksp = self.wksp

        # Get title and detector group info (L1s, Thetas, and Phis)
        title = str(wksp.title)
        groups = wksp.instrument.detector.detector_positions

        # Extract detector group info
        L1 = [float(l1) for (l1, theta, phi) in groups]
        Theta = [float(theta) for (l1, theta, phi) in groups]
        Phi = [float(phi) for (l1, theta, phi) in groups]

        # Get detector group data
        x = np.array(wksp.workspace.axis1.boundaries())
        err_groups = np.array(wksp.workspace.errors)
        y_groups = np.array(wksp.workspace.values()[1])  # 0==error, 1==values

        # Re-sort based on Theta degrees
        err_groups, tmp = self.sort_lists(sorter=Theta, sortee=err_groups)
        y_groups, tmp = self.sort_lists(sorter=Theta, sortee=y_groups)
        L1, tmp = self.sort_lists(sorter=Theta, sortee=L1)
        Phi, tmp = self.sort_lists(sorter=Theta, sortee=Phi)

        # Theta must be sorted last
        Theta, tmp = self.sort_lists(sorter=Theta, sortee=Theta)

        for i, (l1, theta, phi, y, err) in enumerate(
                zip(L1, Theta, Phi, y_groups, err_groups)):
            tag = self.getTag(title)
            info_dict = {
                'tag': tag,
                'L1': l1,
                'Theta': theta,
                'Phi': Phi,
                'yerr': err}
            self.datasets[title] = Dataset(
                x=x, y=y, title=title, info=info_dict)

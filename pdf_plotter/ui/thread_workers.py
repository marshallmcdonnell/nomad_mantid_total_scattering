import os
import collections
import threading
import h5py
import numpy as np

from models \
    import Dataset, CorrectedDatasets, Measurement, Experiment

# -----------------------------------------------------------#
# Measurement-type to workspace-title-startswith Map

mtype2title = collections.OrderedDict()
mtype2title['Sample'] = 'sample'
mtype2title['Container Background'] = 'container_background'
mtype2title['Container'] = 'container'
mtype2title['Vanadium Background'] = 'vanadium_background'
mtype2title['Vanadium'] = 'vanadium'
mtype2title['Correction'] = 'correction'

# workspace-title startswith to Measurement-type Map
title2mtype = collections.OrderedDict()
for k, v in mtype2title.iteritems():
    title2mtype[v] = k


# List of measurement types
mtype_list = [k for k, v in mtype2title.iteritems()]
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
        wksps = [nx[wksp] for wksp in nx
                 if wksp.startswith("mantid_workspace")]

        t_list = list()
        for i, wksp in enumerate(wksps):
            t = DatasetThread(wksp)
            t.corrected_datasets = self.corrected_datasets
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

    # Main thread opens and extracts Nexus and then launchs threads to extract
    # Datasets
    def run(self):
        self.update_status("Loading...")
        self.nxresult = h5py.File(self.f, 'r')
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

    def get_measurement_type(self, title):
        for key in title2mtype:
            if title.startswith(key):
                return title2mtype[key]
        return 'Other'

    def run(self):
        wksp = dict(self.wksp)

        # Get title and detector group info (L1s, Thetas, and Phis)
        title = str(wksp['title'].value[0])
        groups = wksp['instrument']['detector']['detector_positions'].value

        # Extract detector group info
        L1 = [float(l1) for (l1, theta, phi) in groups]
        Theta = [float(theta) for (l1, theta, phi) in groups]
        Phi = [float(phi) for (l1, theta, phi) in groups]

        # Get detector group data
        x = np.array(wksp['workspace']['axis1'].value)
        err_groups = wksp['workspace']['errors'].value
        y_groups = wksp['workspace']['values'].value   # 0==error, 1==values

        # Re-sort based on Theta degrees
        tmp, err_groups = self.sort_lists(sorter=Theta, sortee=err_groups)
        tmp, y_groups = self.sort_lists(sorter=Theta, sortee=y_groups)
        tmp, L1 = self.sort_lists(sorter=Theta, sortee=L1)
        tmp, Phi = self.sort_lists(sorter=Theta, sortee=Phi)

        # Theta must be sorted last
        tmp, Theta = self.sort_lists(sorter=Theta, sortee=Theta)

        dataset_list = list()
        for i, (l1, theta, phi, y, err) in enumerate(
                zip(L1, Theta, Phi, y_groups, err_groups)):
            info_dict = {
                'L1': l1,
                'Theta': theta,
                'Phi': Phi,
                'yerr': err}
            dataset_title = "Bank: {0:.2f}".format(theta)
            dataset_list.append(
                Dataset(x=x, y=y,
                        title=dataset_title,
                        info=info_dict
                        )
            )

        # Get Measurement-type based on title
        measurement_type = self.get_measurement_type(title)
        info_dict = {'measurement_type': measurement_type}
        self.corrected_datasets[title] = CorrectedDatasets(
            datasets=dataset_list,
            title=title,
            info=info_dict
        )
        return


# -----------------------------------------------------------------#
# Thread to handle creating Measurement from CoorectedDatasets
class MeasurementThread(threading.Thread):
    def __init__(self, measurement_type):
        self.measurement_type = measurement_type
        threading.Thread.__init__(self)

    def get_measurement_of_type(self, my_type):
        cd_list = [cd for title, cd in self.corrected_datasets.items()
                   if cd.info['measurement_type'] == my_type]
        self.measurements[my_type] = Measurement(corrected_datasets=cd_list,
                                                 title=my_type)

    def get_other_measurement(self):
        cd_list = [cd for title, cd in self.corrected_datasets.items()
                   if cd.info['measurement_type'] not in mtype_list]

        if len(cd_list) == 0:
            return

        self.measurements['Other'] = Measurement(corrected_datasets=cd_list,
                                                 title='Other')

    def run(self):
        if self.measurement_type == 'Other':
            self.get_other_measurement()
        else:
            self.get_measurement_of_type(self.measurement_type)

# -----------------------------------------------------------------#
# Thread to handle creating the Experiment from CorrectedDatasets


class ExperimentThread(threading.Thread):
    def __init__(self):
        self.experiment = None
        self.measurements = dict()
        threading.Thread.__init__(self)

    # Multithreaded extraction of Datasets from Nexus file
    def create_measurements(self):
        my_list = mtype_list + ['Other']
        t_list = list()
        for i, measurement_type in enumerate(my_list):
            t = MeasurementThread(measurement_type)
            t.corrected_datasets = self.corrected_datasets
            t.measurements = self.measurements
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

    def create_experiments(self):
        measurement_list = list()
        for title, measurement in self.measurements.iteritems():
            measurement_list.extend([measurement])
        name, ext = os.path.splitext(self.filename)
        name = os.path.basename(name)
        self.experiment = Experiment(measurements=measurement_list, title=name)

    # Main thread opens and extracts Nexus and then launchs threads to extract
    # Datasets
    def run(self):
        self.update_status("Creating Measurements...")
        self.create_measurements()
        self.update_status("Done with Measurements!")
        self.update_status("Creating Experiment...")
        self.create_experiments()
        self.update_status("Done loading Experiment!")
        self.update_experiment(self.experiment)
        return

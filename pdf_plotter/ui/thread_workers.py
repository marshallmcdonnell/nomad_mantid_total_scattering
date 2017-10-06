
import threading
import numpy as np
import h5py

from models \
    import Dataset, CorrectedDatasets, Measurement, Experiment


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
        wksps = [nx[wksp] for wksp in nx 
                          if  wksp.startswith("mantid_workspace")]

        t_list = list()
        for i, wksp in enumerate(wksps):
            t = DatasetThread(wksp)
            t.corrected_datasets = self.corrected_datasets
            t.start()
            t_list.append(t)

        for t in t_list:
            t.join()

    # Main thread opens and extracts Nexus and then launchs threads to extract
    # Datasets
    def run(self):
        self.update_status("Loading...")
        self.nxresult =  h5py.File(self.f, 'r')
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
        elif title.startswith('vanadium_background'):
            return 'Vanadium Background'
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
        wksp = dict(self.wksp)

        # Get title and detector group info (L1s, Thetas, and Phis)
        title = str(wksp['title'].value[0])
        print title
        groups = wksp['instrument']['detector']['detector_positions'].value

        # Extract detector group info
        L1 = [float(l1) for (l1, theta, phi) in groups]
        Theta = [float(theta) for (l1, theta, phi) in groups]
        Phi = [float(phi) for (l1, theta, phi) in groups]

        # Get detector group data
        x = np.array(wksp['workspace']['axis1'].value)
        err_groups = wksp['workspace']['errors'].value
        y_groups =  wksp['workspace']['values'].value   # 0==error, 1==values

        # Re-sort based on Theta degrees
        tmp, err_groups = self.sort_lists(sorter=Theta, sortee=err_groups)
        tmp, y_groups   = self.sort_lists(sorter=Theta, sortee=y_groups)
        tmp, L1         = self.sort_lists(sorter=Theta, sortee=L1)
        tmp, Phi        = self.sort_lists(sorter=Theta, sortee=Phi)

        # Theta must be sorted last
        tmp, Theta      = self.sort_lists(sorter=Theta, sortee=Theta)

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

        # Get CorrectedDatasets type based on title (called the Tag)
        tag = self.getTag(title)
        info_dict = {'tag': tag}
        self.corrected_datasets[title] = CorrectedDatasets( 
                                            datasets = dataset_list,
                                            title = title,
                                            info = info_dict
                                         )


# -----------------------------------------------------------------#
# Thread to handle creating Measurement from CoorectedDatasets
class MeasurementThread(threading.Thread):
    def __init__(self, measurement_type):
        self.measurement_type = measurement_type
        threading.Thread.__init__(self)


    def get_measurement(self):
        cd_list = list()
        for title, cd in self.corrected_datasets.items():
            tag = cd.info['tag']
            if tag == self.measurement_type:
                cd_list.append(cd)

        self.measurements[tag] = Measurement( corrected_datasets=cd_list,
                                              title=tag)
    def get_other_measurement(self):
        pass
                                              
    def run(self):
        if self.measurement_type == 'Other':
            self.get_other_measurement()
        else:
            self.get_measurement()
               
# -----------------------------------------------------------------#
# Thread to handle creating the Experiment from CorrectedDatasets

class ExperimentThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

 
    # Multithreaded extraction of Datasets from Nexus file
    def create_measurements(self):

        t_list = list()
        for i, wksp in enumerate(wksps):
            t = MeasurementThread(wksp)
            t.corrected_datasets = self.corrected_datasets
            t.start()
            t_list.append(t)

        for t in t_list:
            t.join()

    # Main thread opens and extracts Nexus and then launchs threads to extract
    # Datasets
    def run(self):
        self.update_status("Creating Measurements...")
        self.create_measurements()
        self.update_status("Done with Measurements!")
        self.update_status("Creating Experiment...")
        self.create_experiments()
        self.update_status("Done with Experiment!")
        return

   

 

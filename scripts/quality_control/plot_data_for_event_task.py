import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.functions.visualize_ecg import plot_heartbeat_on_ecg, \
    convert_from_mat_format
from scripts.functions.utils import import_variables_from_yaml
import glob
import scipy.io

variables = import_variables_from_yaml("configuration/configuration.yaml")

mat_folder = variables['EVENT_MAT_FOLDER']
mat_files = glob.glob(os.path.join(mat_folder, "*.mat"))


patients_list = []
ecgs_list = []
leads_list = []
labels_event_list = []
signals_list = []
sets_list = []

for mat_file in mat_files:

    # Load each file content into data
    data = scipy.io.loadmat(mat_file)

    patients_list, ecgs_list, labels_event_list, \
        sets_list, leads_list, signals_list = convert_from_mat_format(
            data, patients_list, ecgs_list, labels_event_list,
            sets_list, leads_list, signals_list)

tot_len = len(patients_list)

for idx, (patient,
          ecg,
          lead,
          signal,
          label) in enumerate(zip(patients_list,
                                  ecgs_list,
                                  leads_list,
                                  signals_list,
                                  labels_event_list)):

    plot_heartbeat_on_ecg(signal, lead, patient, label, idx, tot_len)

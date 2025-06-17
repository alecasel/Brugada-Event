import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.functions.utils import import_variables_from_yaml
from scripts.functions.model import create_student_model, build_supervised_model
from scripts.functions.prediction import \
    process_folder, plot_umap_patient_clusters

variables = import_variables_from_yaml("configuration/configuration.yaml")

xml_syncope_directory = variables['XML_SYNCOPE_DIRECTORY']
xml_asymptomatic_directory = variables['XML_ASYMPTOMATIC_DIRECTORY']
xml_event_directory = variables['XML_EVENT_DIRECTORY']
xml_all_directory = variables['XML_ALL_DIRECTORY']

test_output_folder = variables['TEST_OUTPUT_FOLDER']

student_model_path = variables['WEIGHTS_STUDENT_PATH']
student_model = create_student_model(851, 1)
student_model.load_weights(student_model_path)

supervised_model_path = variables["WEIGHTS_SUPERVISED_3_CLASSES_PATH"]
supervised_model = build_supervised_model(851, 1)
supervised_model.load_weights(supervised_model_path)

process_folder(xml_all_directory,
               "scripts/tests/output/supervised_event_predictions.xlsx",
               supervised_model,
               consider_embeddings=False,
               leads=['V1', 'V2', 'V3'],
               excluded_folders=['SINCOPI', 'SINCOPI VAGALI', 'ASINTOMATICI'],
               num_patients=30000000)

# plot_umap_patient_clusters(Z_student, patient_ids, classes)

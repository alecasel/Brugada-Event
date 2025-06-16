import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.functions.utils import import_variables_from_yaml
from scripts.functions.prediction import process_folder
from scripts.functions.model import build_supervised_model, \
    create_student_model

variables = import_variables_from_yaml("configuration/configuration.yaml")

xml_data_directory = variables['XML_SYNCOPE_DIRECTORY']
test_output_folder = variables['TEST_OUTPUT_FOLDER']
supervised_model_path = variables['WEIGHTS_SUPERVISED_3_CLASSES_PATH']
student_model_path = variables['WEIGHTS_STUDENT_PATH']

supervised_model = build_supervised_model(851, 1)
supervised_model.load_weights(supervised_model_path)
student_model = create_student_model(851, 1)
student_model.load_weights(student_model_path)

# Process xml files with the supervised model
print("Processing XML files with the supervised model...")
process_folder(xml_data_directory,
               os.path.join(test_output_folder, 'supervised_predictions.xlsx'),
               supervised_model)

# Process xml files with the student model
print("\nProcessing XML files with the student model...")
process_folder(xml_data_directory,
               os.path.join(test_output_folder, 'student_predictions.xlsx'),
               student_model)

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.functions.utils import import_variables_from_yaml
from scripts.functions.model import build_supervised_model
from scripts.functions.prediction import process_json

variables = import_variables_from_yaml("configuration/configuration.yaml")

json_path = variables['JSON_GAITA_PATH']

supervised_model_path = variables["WEIGHTS_SUPERVISED_3_CLASSES_PATH"]
supervised_model = build_supervised_model(851, 1)
supervised_model.load_weights(supervised_model_path)

process_json(json_path,
             r'scripts\tests\output\supervised_gaita_ecgs_predictions.xlsx',
             supervised_model,
             leads=['V1', 'V2', 'V3'])

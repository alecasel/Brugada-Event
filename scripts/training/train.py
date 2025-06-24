import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.functions.model import build_risk_stratification_model, \
    compile_risk_model
from scripts.functions.utils import set_reproducible_config


set_reproducible_config()

leads = ['V1', 'V2', 'V3', 'II', 'aVL', 'aVF']

risk_model = build_risk_stratification_model(seq_length=851,
                                             num_leads=6,
                                             unified_approach=False,
                                             risk_output_type='classification')
compile_risk_model(risk_model)

for layer in risk_model.layers:
    weights = layer.get_weights()
    print(f"{layer.name}: {'✔️' if weights else '❌'}")

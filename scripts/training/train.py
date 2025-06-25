from scripts.functions.utils import import_variables_from_yaml, \
    set_reproducible_config
from scripts.functions.model import build_risk_stratification_model, \
    compile_risk_model
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from scripts.functions.dataset import select_data, prepare_brugada_dataset
import glob
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


set_reproducible_config()

leads_of_interest = ['V1', 'V2', 'V3', 'II', 'III', 'aVF']
leads_to_invert = ['II', 'III', 'aVF']

risk_model = build_risk_stratification_model(seq_length=851,
                                             num_leads=6,
                                             unified_approach=False,
                                             risk_output_type='classification')
compile_risk_model(risk_model)

for layer in risk_model.layers:
    weights = layer.get_weights()
    print(f"{layer.name}: {'✔️' if weights else '❌'}")


variables = import_variables_from_yaml("configuration/configuration.yaml")

mat_folder = variables['EVENT_MAT_FOLDER']
mat_files = glob.glob(os.path.join(mat_folder, "*.mat"))

patients_list, ecgs_list, labels_event_list, \
    sets_list, leads_list, signals_list = prepare_brugada_dataset(
        mat_files)

patients_list, ecgs_list, labels_event_list, \
    sets_list, leads_list, signals_list = select_data(
        patients_list,
        ecgs_list,
        labels_event_list,
        sets_list,
        leads_list,
        signals_list,
        leads_of_interest=leads_of_interest,
        leads_to_invert=leads_to_invert)

# Crea una lista di tuple per raggruppare i dati
data = list(zip(patients_list, ecgs_list, labels_event_list,
                sets_list, leads_list, signals_list))

# Filtra solo i dati con label_event validi
filtered_data = [d for d in data if d[2] in ("EVENT", "NO EVENT")]

# Suddividi i dati in base al vincolo di set
trainval_data = [d for d in filtered_data if d[3] in ("train/val",
                                                      "train/val/test")]
test_data = [d for d in filtered_data if d[3] == "train/val/test"]

# Conta le classi per train/val
labels_trainval = [d[2] for d in trainval_data]
counter_trainval = Counter(labels_trainval)
min_class_trainval = min(counter_trainval.values())

# Bilancia il dataset train/val 50-50 tra EVENT e NO EVENT
event_data_trainval = [d for d in trainval_data if d[2] == "EVENT"]
no_event_data_trainval = [d for d in trainval_data if d[2] == "NO EVENT"]

np.random.seed(42)
event_data_trainval = np.random.choice(
    event_data_trainval, min_class_trainval, replace=False)
no_event_data_trainval = np.random.choice(
    no_event_data_trainval, min_class_trainval, replace=False)

balanced_trainval_data = np.concatenate(
    [event_data_trainval, no_event_data_trainval])
np.random.shuffle(balanced_trainval_data)

# Split train/val 82.35%-17.65% (per ottenere 70-15 su totale)
train_size = int(0.8235 * len(balanced_trainval_data))
train_data = balanced_trainval_data[:train_size]
val_data = balanced_trainval_data[train_size:]

# Bilancia anche il test set (se vuoi)
labels_test = [d[2] for d in test_data]
counter_test = Counter(labels_test)
min_class_test = min(counter_test.values()) if counter_test else 0

if min_class_test > 0:
    event_data_test = [d for d in test_data if d[2] == "EVENT"]
    no_event_data_test = [d for d in test_data if d[2] == "NO EVENT"]
    event_data_test = np.random.choice(
        event_data_test, min_class_test, replace=False)
    no_event_data_test = np.random.choice(
        no_event_data_test, min_class_test, replace=False)
    balanced_test_data = np.concatenate([event_data_test, no_event_data_test])
    np.random.shuffle(balanced_test_data)
    test_data = balanced_test_data
else:
    test_data = np.array([])


# Separa le liste
def unpack(data):
    if len(data) == 0:
        return [], [], [], [], [], []
    return [list(x) for x in zip(*data)]


train_patients, train_ecgs, train_labels, \
    train_sets, train_leads, train_signals = unpack(train_data)
val_patients, val_ecgs, val_labels, \
    val_sets, val_leads, val_signals = unpack(val_data)
test_patients, test_ecgs, test_labels, \
    test_sets, test_leads, test_signals = unpack(test_data)

print(f"Train: {Counter(train_labels)}")
print(f"Val: {Counter(val_labels)}")
print(f"Test: {Counter(test_labels)}")

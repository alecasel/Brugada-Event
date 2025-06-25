import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.utils import class_weight
import numpy as np
from scripts.functions.utils import import_variables_from_yaml, \
    set_reproducible_config
from scripts.functions.model import build_risk_stratification_model, \
    compile_risk_model
from collections import Counter
from scripts.functions.dataset import fix_existing_data_constraints, \
    select_data, prepare_brugada_dataset, unpack_data, split_data
import glob
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder


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

fixed_data = fix_existing_data_constraints(data)
train_data, val_data, test_data = split_data(fixed_data)

train_patients, train_ecgs, train_labels, \
    train_sets, train_leads, train_signals = unpack_data(train_data)
val_patients, val_ecgs, val_labels, \
    val_sets, val_leads, val_signals = unpack_data(val_data)

print(f"Train: {Counter(train_labels)}")
print(f"Val: {Counter(val_labels)}")

class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.25, patience=7, min_lr=1e-7)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True)

X_train = np.expand_dims(train_signals, axis=-1)
X_valid = np.expand_dims(val_signals, axis=-1)
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(train_labels)
Y_valid = label_encoder.transform(val_labels)

history = risk_model.fit(
    X_train, Y_train,
    validation_data=(X_valid, Y_valid),
    epochs=1,
    verbose=1,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

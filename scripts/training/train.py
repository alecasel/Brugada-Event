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
from scripts.functions.dataset import organize_ecg_data_for_multihead, \
    print_data_summary, select_data, prepare_brugada_dataset, unpack_data, \
    split_data
import glob
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-7,
                    help='Learning rate')
parser.add_argument('--lr_patience', type=int, default=5,
                    help='Learning rate patience')
parser.add_argument('--lr_min', type=float, default=1e-6,
                    help='Min Learning rate')
parser.add_argument('--patience', type=int, default=20,
                    help='Patience')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--leads', nargs='+', type=str,
                    default=['V1', 'V2', 'V3', 'II', 'III', 'aVF'],
                    help='Leads of Interest')
parser.add_argument('--cross_attention_heads', type=int, default=3,
                    help='Cross Attention Heads')
parser.add_argument('--history_name', type=str,
                    default='history_risk_model.npz', help='History filename')
parser.add_argument('--children', nargs='+', type=str,
                    default=[''],
                    help='Children: to exclude from the analysis')
parser.add_argument('--no_type_1', nargs='+', type=str,
                    default=[''],
                    help='Patients without BrP type 1: ' +
                         'to exclude from the analysis')

args = parser.parse_args()


set_reproducible_config()

leads_of_interest = args.leads
leads_to_invert = [lead for lead in leads_of_interest
                   if not lead.startswith('V') and lead != 'aVR']

risk_model = build_risk_stratification_model(
    seq_length=851, num_leads=len(leads_of_interest),
    num_heads_cross_attention=args.cross_attention_heads,
    unified_approach=False, risk_output_type='probability')

compile_risk_model(risk_model, args.lr)

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
        leads_to_invert=leads_to_invert,
        patients_to_exclude=args.children + args.no_type_1
    )

# Crea una lista di tuple per raggruppare i dati
data = list(zip(patients_list, ecgs_list, labels_event_list,
                sets_list, leads_list, signals_list))

train_data, val_data, test_data = split_data(data)

train_patients, train_ecgs, train_labels, \
    train_sets, train_leads, train_signals = unpack_data(train_data)
val_patients, val_ecgs, val_labels, \
    val_sets, val_leads, val_signals = unpack_data(val_data)

print(f"Train: {Counter(train_labels)}")
print(f"Val: {Counter(val_labels)}")

class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

# Applica la trasformazione
print("Elaborazione training set...")
X_train, Y_train, train_sample_info = organize_ecg_data_for_multihead(
    train_patients, train_ecgs, train_signals, train_leads, train_labels,
    leads_of_interest)

print("\nElaborazione validation set...")
X_valid, Y_valid, val_sample_info = organize_ecg_data_for_multihead(
    val_patients, val_ecgs, val_signals, val_leads, val_labels,
    leads_of_interest)

# Encoding delle labels assicurando che la classe negativa ("NO") sia 0
label_encoder = LabelEncoder()
# Ordina le classi in modo che quella che contiene "NO" sia la prima
classes = sorted(set(Y_train), key=lambda x: (not ("NO" in x), x))
label_encoder.fit(classes)
Y_train_encoded = label_encoder.transform(Y_train)
Y_valid_encoded = label_encoder.transform(Y_valid)

# Stampa riassunti
print_data_summary(X_train, Y_train, train_sample_info, "Training")
print_data_summary(X_valid, Y_valid, val_sample_info, "Validation")

print("\nLabel encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {i}")

# Verifica finale
print("\nFinal shapes:")
print(f"X_train: {X_train.shape}")  # (n_samples, seq_length, 12)
print(f"X_valid: {X_valid.shape}")
print(f"Y_train_encoded: {Y_train_encoded.shape}")
print(f"Y_valid_encoded: {Y_valid_encoded.shape}")

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.25,
                              patience=args.lr_patience,
                              min_lr=args.lr_min)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=args.patience,
                               restore_best_weights=True)

checkpoint_loss = ModelCheckpoint(
    'best_weights_val_loss.h5', monitor='val_loss',
    save_best_only=True, mode='min', verbose=1
)

checkpoint_auc = ModelCheckpoint(
    'best_weights_val_auc.h5', monitor='val_auc',
    save_best_only=True, mode='max', verbose=1
)

checkpoint_acc = ModelCheckpoint(
    'best_weights_val_accuracy.h5', monitor='val_binary_accuracy',
    save_best_only=True, mode='max', verbose=1
)

history = risk_model.fit(
    X_train, Y_train_encoded,
    validation_data=(X_valid, Y_valid_encoded),
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=[early_stopping, reduce_lr, checkpoint_loss,
               checkpoint_auc, checkpoint_acc],
    class_weight=class_weights
)

test_output_folder = variables["TEST_OUTPUT_FOLDER"]

os.makedirs(test_output_folder, exist_ok=True)

np.savez_compressed(
    os.path.join(test_output_folder, args.history_name),
    **history.history)

def preprocess_ecg_data(ecg_files):
    """
    """

    from scripts.functions.ecg_extraction import import_ecg_data, lowpass
    # TODO


def prepare_brugada_dataset(ecg_files,
                            clinical_data,
                            test_size=0.15,
                            val_size=0.15):
    """
    Prepara il dataset per il training del modello
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Carica e preprocessa ECG
    X = preprocess_ecg_data(ecg_files)

    # Definisci etichette di rischio
    y = "TODO"

    # Associa ogni dato a un paziente
    # Si assume che ecg_files o clinical_data contenga una colonna/chiave 'patient_id'

    # Estrai lista unica dei pazienti
    if isinstance(ecg_files, pd.DataFrame):
        patient_ids = ecg_files['patient_id'].unique()
    else:
        # Se ecg_files è una lista di dict o altro formato
        patient_ids = np.unique([x['patient_id'] for x in ecg_files])

    # Split dei pazienti
    patients_train, patients_temp = train_test_split(
        patient_ids, test_size=(val_size + test_size), random_state=42
    )
    relative_val_size = val_size / (val_size + test_size)
    patients_val, patients_test = train_test_split(
        patients_temp, test_size=1 - relative_val_size, random_state=42
    )

    # Crea maschere per assegnare i dati ai rispettivi split
    def mask_by_patients(data, patients):
        if isinstance(data, pd.DataFrame):
            return data[data['patient_id'].isin(patients)]
        else:
            return [x for x in data if x['patient_id'] in patients]

    X_train = mask_by_patients(X, patients_train)
    X_val = mask_by_patients(X, patients_val)
    X_test = mask_by_patients(X, patients_test)

    y_train = mask_by_patients(y, patients_train)
    y_val = mask_by_patients(y, patients_val)
    y_test = mask_by_patients(y, patients_test)

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

import numpy as np
from collections import Counter


def prepare_brugada_dataset(mat_files):
    """
    Prepara il dataset per il training del modello
    """
    import scipy
    from scripts.functions.visualize_ecg import convert_from_mat_format

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

    return patients_list, ecgs_list, labels_event_list, \
        sets_list, leads_list, signals_list


def select_data(patients_list,
                ecgs_list,
                labels_event_list,
                sets_list,
                leads_list,
                signals_list,
                leads_of_interest,
                leads_to_invert):
    """
    """

    new_patients_list = []
    new_ecgs_list = []
    new_labels_event_list = []
    new_sets_list = []
    new_leads_list = []
    new_signals_list = []

    for i, lead in enumerate(leads_list):
        if lead in leads_of_interest:
            signal = signals_list[i]
            if lead in leads_to_invert:
                inverted_signal = -signal
                baseline_shift = signal.mean() + inverted_signal.mean()
                signal = inverted_signal - baseline_shift

            new_patients_list.append(patients_list[i])
            new_ecgs_list.append(ecgs_list[i])
            new_labels_event_list.append(labels_event_list[i])
            new_sets_list.append(sets_list[i])
            new_leads_list.append(lead)
            new_signals_list.append(signal)

    return new_patients_list, new_ecgs_list, new_labels_event_list, \
        new_sets_list, new_leads_list, new_signals_list


def split_data(data):
    """
    Split data into train, validation, and test sets ensuring that:
    - Each patient appears in only one set
    - Each ECG appears in only one set
    - All derivations of the same ECG stay together
    - Balanced distribution of patients and classes across sets
    """
    from collections import defaultdict

    # Filtra solo i dati con label_event validi
    filtered_data = [d for d in data if d[2] in ("EVENT", "NO EVENT")]

    # Diagnostica dei dati originali
    print("=== DIAGNOSTICA DATI ORIGINALI ===")
    original_labels = [d[2] for d in filtered_data]
    original_counter = Counter(original_labels)
    print(f"Distribuzione classi totale: {dict(original_counter)}")
    print(f"Totale campioni: {len(filtered_data)}")

    # Raggruppa per paziente e ECG
    patient_ecg_groups = defaultdict(list)
    for sample in filtered_data:
        patient_id = sample[0]
        ecg_id = sample[1]
        key = (patient_id, ecg_id)
        patient_ecg_groups[key].append(sample)

    print(
        f"Numero totale di pazienti: {len(set(d[0] for d in filtered_data))}")
    print(f"Numero totale di ECG: {len(patient_ecg_groups)}")
    print("Media derivazioni per ECG: "
          f"{len(filtered_data) / len(patient_ecg_groups):.1f}")

    # Analizza per PAZIENTE (non per ECG) - raggruppa tutto per paziente
    patient_info = defaultdict(
        lambda: {'ecgs': [], 'constraints': set(), 'labels': []})

    for (patient_id, ecg_id), samples in patient_ecg_groups.items():
        patient_info[patient_id]['ecgs'].append((patient_id, ecg_id))
        patient_info[patient_id]['constraints'].update(
            sample[3] for sample in samples)
        patient_info[patient_id]['labels'].extend(
            sample[2] for sample in samples)

    # Analizza constraint per paziente
    patient_constraints = {}
    for patient_id, info in patient_info.items():
        constraints = info['constraints']
        if len(constraints) == 1:
            patient_constraints[patient_id] = list(constraints)[0]
        else:
            # Paziente con constraint misti - prendi il più restrittivo
            if "train/val/test" in constraints:
                # Può andare ovunque
                patient_constraints[patient_id] = "train/val/test"
            else:
                patient_constraints[patient_id] = "train/val"

    print("\nAnalisi constraint per paziente:")
    constraint_counts = Counter(patient_constraints.values())
    for constraint, count in constraint_counts.items():
        print(f"  {constraint}: {count} pazienti")

    # Analizza distribuzione classi per paziente
    patient_class_distribution = {}
    for patient_id, info in patient_info.items():
        labels = info['labels']
        event_count = labels.count('EVENT')
        no_event_count = labels.count('NO EVENT')

        # Determina la "classe prevalente" del paziente
        if event_count > no_event_count:
            patient_class_distribution[patient_id] = 'EVENT'
        elif no_event_count > event_count:
            patient_class_distribution[patient_id] = 'NO EVENT'
        else:
            patient_class_distribution[patient_id] = 'MIXED'

    print("\nDistribuzione classi per paziente:")
    class_counts = Counter(patient_class_distribution.values())
    for class_type, count in class_counts.items():
        print(f"  {class_type}: {count} pazienti")

    # Strategia: ignoriamo i constraint originali
    # e facciamo uno split bilanciato
    # Raggruppiamo i pazienti per classe prevalente
    event_patients = [
        pid for pid, cls in patient_class_distribution.items()
        if cls == 'EVENT']
    no_event_patients = [
        pid for pid, cls in patient_class_distribution.items()
        if cls == 'NO EVENT']
    mixed_patients = [
        pid for pid, cls in patient_class_distribution.items()
        if cls == 'MIXED']

    print("\nPazienti per classe prevalente:")
    print(f"  EVENT: {len(event_patients)}")
    print(f"  NO EVENT: {len(no_event_patients)}")
    print(f"  MIXED: {len(mixed_patients)}")

    # Split bilanciato: 70% train, 15% val, 15% test
    # per ogni gruppo di pazienti
    np.random.seed(42)

    def split_patients_proportionally(patients_list,
                                      train_ratio=0.70,
                                      val_ratio=0.15):
        """Split una lista di pazienti mantenendo le proporzioni"""
        if not patients_list:
            return [], [], []

        shuffled = patients_list.copy()
        np.random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_patients = shuffled[:n_train]
        val_patients = shuffled[n_train:n_train + n_val]
        test_patients = shuffled[n_train + n_val:]

        return train_patients, val_patients, test_patients

    # Split per ogni gruppo di classe
    event_train, event_val, event_test = split_patients_proportionally(
        event_patients)
    no_event_train, no_event_val, no_event_test = \
        split_patients_proportionally(no_event_patients)
    mixed_train, mixed_val, mixed_test = split_patients_proportionally(
        mixed_patients)

    # Combina i risultati
    train_patients = event_train + no_event_train + mixed_train
    val_patients = event_val + no_event_val + mixed_val
    test_patients = event_test + no_event_test + mixed_test

    print("\nSplit finale pazienti:")
    print(f"Train: {len(train_patients)} (EVENT: {len(event_train)}, "
          f"NO EVENT: {len(no_event_train)}, MIXED: {len(mixed_train)})")
    print(f"Val: {len(val_patients)} (EVENT: {len(event_val)}, "
          f"NO EVENT: {len(no_event_val)}, MIXED: {len(mixed_val)})")
    print(f"Test: {len(test_patients)} (EVENT: {len(event_test)}, "
          f"NO EVENT: {len(no_event_test)}, MIXED: {len(mixed_test)})")

    # Verifica che non ci siano sovrapposizioni
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("❌ ERRORE: Sovrapposizioni trovate nella divisione pazienti!")
        return [], [], []

    # Costruisci i dataset finali raccogliendo tutti gli ECG per ogni paziente
    train_data = []
    val_data = []
    test_data = []

    for patient_id in train_patients:
        for ecg_key in patient_info[patient_id]['ecgs']:
            train_data.extend(patient_ecg_groups[ecg_key])

    for patient_id in val_patients:
        for ecg_key in patient_info[patient_id]['ecgs']:
            val_data.extend(patient_ecg_groups[ecg_key])

    for patient_id in test_patients:
        for ecg_key in patient_info[patient_id]['ecgs']:
            test_data.extend(patient_ecg_groups[ecg_key])

    print("\n=== RISULTATI FINALI ===")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Conta ECG per set
    train_ecg_count = len(set((d[0], d[1]) for d in train_data))
    val_ecg_count = len(set((d[0], d[1]) for d in val_data))
    test_ecg_count = len(set((d[0], d[1]) for d in test_data))

    print(f"Train ECG: {train_ecg_count}")
    print(f"Val ECG: {val_ecg_count}")
    print(f"Test ECG: {test_ecg_count}")

    # Verifica distribuzione classi
    train_labels = [d[2] for d in train_data]
    val_labels = [d[2] for d in val_data]
    test_labels = [d[2] for d in test_data]

    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    test_counter = Counter(test_labels)

    print("\nDistribuzione classi:")
    print(f"Train: {dict(train_counter)}")
    print(f"Val: {dict(val_counter)}")
    print(f"Test: {dict(test_counter)}")

    # Verifica pazienti unici finali
    train_patients_final = set(d[0] for d in train_data)
    val_patients_final = set(d[0] for d in val_data)
    test_patients_final = set(d[0] for d in test_data)

    print("\nPazienti unici finali:")
    print(f"Train: {len(train_patients_final)}")
    print(f"Val: {len(val_patients_final)}")
    print(f"Test: {len(test_patients_final)}")

    # Verifica sovrapposizioni pazienti
    train_val_final_overlap = train_patients_final.intersection(
        val_patients_final)
    train_test_final_overlap = train_patients_final.intersection(
        test_patients_final)
    val_test_final_overlap = val_patients_final.intersection(
        test_patients_final)

    print("\nSovrapposizioni pazienti:")
    print(
        f"Train-Val: {len(train_val_final_overlap)} "
        f"{'✅' if len(train_val_final_overlap) == 0 else '❌'}")
    print(
        f"Train-Test: {len(train_test_final_overlap)} "
        f"{'✅' if len(train_test_final_overlap) == 0 else '❌'}")
    print(
        f"Val-Test: {len(val_test_final_overlap)} "
        f"{'✅' if len(val_test_final_overlap) == 0 else '❌'}")

    # Verifica ECG unici
    train_ecg_ids = set((d[0], d[1]) for d in train_data)
    val_ecg_ids = set((d[0], d[1]) for d in val_data)
    test_ecg_ids = set((d[0], d[1]) for d in test_data)

    train_val_ecg_overlap = train_ecg_ids.intersection(val_ecg_ids)
    train_test_ecg_overlap = train_ecg_ids.intersection(test_ecg_ids)
    val_test_ecg_overlap = val_ecg_ids.intersection(test_ecg_ids)

    print("\nSovrapposizioni ECG:")
    print(
        f"Train-Val: {len(train_val_ecg_overlap)} "
        f"{'✅' if len(train_val_ecg_overlap) == 0 else '❌'}")
    print(
        f"Train-Test: {len(train_test_ecg_overlap)} "
        f"{'✅' if len(train_test_ecg_overlap) == 0 else '❌'}")
    print(
        f"Val-Test: {len(val_test_ecg_overlap)} "
        f"{'✅' if len(val_test_ecg_overlap) == 0 else '❌'}")

    # Verifica conservazione dati
    total_final = len(train_data) + len(val_data) + len(test_data)
    print("\nVerifica conservazione dati:")
    print(f"Dati originali filtrati: {len(filtered_data)}")
    print(f"Dati finali totali: {total_final}")
    print(
        "Dati conservati: "
        f"{'✅ SÌ' if total_final == len(filtered_data) else '❌ NO'}")

    return train_data, val_data, test_data


def unpack_data(data):
    if len(data) == 0:
        return [], [], [], [], [], []
    return [list(x) for x in zip(*data)]


def organize_ecg_data_for_multihead(patients_list,
                                    ecgs_list,
                                    signals_list,
                                    leads_list,
                                    labels_list,
                                    leads_of_interest):
    """
    Organizza i dati ECG per il modello multi-head attention.
    Ogni finestra temporale di 851ms diventa un campione separato.
    """

    # Raggruppa per (paziente, ECG, window_id)
    ecg_data = {}
    # Tiene traccia del numero di finestre per (paziente, ecg)
    window_counters = {}

    for patient, ecg, signal, lead, label in zip(patients_list, ecgs_list,
                                                 signals_list, leads_list,
                                                 labels_list):

        # Trova quale finestra è questa per questa combinazione
        # (paziente, ecg, lead)
        lead_key = (patient, ecg, lead)
        if lead_key not in window_counters:
            window_counters[lead_key] = 0
        else:
            window_counters[lead_key] += 1

        window_id = window_counters[lead_key]

        # Chiave finale include window_id
        key = (patient, ecg, window_id)

        if key not in ecg_data:
            ecg_data[key] = {
                'signals': {},
                'label': label,
                'patient': patient,
                'ecg': ecg,
                'window_id': window_id
            }

        ecg_data[key]['signals'][lead] = signal

    # Resto della funzione uguale...
    all_signals = []
    for data in ecg_data.values():
        all_signals.extend(data['signals'].values())
    max_length = max(len(signal) for signal in all_signals)

    print(f"Numero totale finestre: {len(ecg_data)}")
    print(f"Lunghezza segnale: {max_length}")

    # Costruisci arrays finali
    n_samples = len(ecg_data)
    X = np.zeros((n_samples, max_length, len(leads_of_interest)))
    y = []
    sample_info = []

    for i, (key, data) in enumerate(ecg_data.items()):
        patient, ecg, window_id = key
        y.append(data['label'])
        sample_info.append({
            'patient': patient,
            'ecg': ecg,
            'window_id': window_id
        })

        # Riempi le derivazioni disponibili
        for lead, signal in data['signals'].items():
            if lead in leads_of_interest:
                lead_idx = leads_of_interest.index(lead)
                padded_signal = np.pad(signal,
                                       (0, max_length - len(signal)),
                                       mode='constant', constant_values=0)
                X[i, :, lead_idx] = padded_signal

    return X, np.array(y), sample_info


def print_data_summary(X,
                       y,
                       sample_info,
                       dataset_name="Dataset"):
    """Stampa riassunto del dataset"""

    print(f"\n=== {dataset_name} Summary ===")
    print(f"Shape: {X.shape}")
    print(f"Labels distribution: {Counter(y)}")

    # Conta derivazioni non-zero per campione
    non_zero_leads = []
    for i in range(X.shape[0]):
        # Conta derivazioni che hanno almeno un valore non-zero
        leads_with_data = np.sum(np.any(X[i] != 0, axis=0))
        non_zero_leads.append(leads_with_data)

    print(f"Derivazioni per campione - Min: {min(non_zero_leads)}, "
          f"Max: {max(non_zero_leads)}, Media: {np.mean(non_zero_leads):.1f}")

    # Mostra alcuni esempi
    print("Primi 5 campioni:")
    for i in range(min(5, len(sample_info))):
        info = sample_info[i]
        leads_count = non_zero_leads[i]
        print(f"  Sample {i}: Patient {info['patient']}, ECG {info['ecg']}, "
              f"{leads_count} derivazioni, Label: {y[i]}")

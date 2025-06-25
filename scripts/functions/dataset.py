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

    for i, lead in enumerate(leads_list):
        if lead not in leads_of_interest:
            patients_list.pop(i)
            ecgs_list.pop(i)
            labels_event_list.pop(i)
            sets_list.pop(i)
            leads_list.pop(i)
            signals_list.pop(i)
        else:
            if lead in leads_to_invert:
                signal = signals_list[i]
                inverted_signal = -signal
                baseline_shift = signal.mean() + inverted_signal.mean()
                signals_list[i] = inverted_signal - baseline_shift

    return patients_list, ecgs_list, labels_event_list, \
        sets_list, leads_list, signals_list


def split_data(data):
    """
    Split data into train, validation, and test sets with balanced classes.
    """
    from collections import Counter
    import numpy as np

    # Filtra solo i dati con label_event validi
    filtered_data = [d for d in data if d[2] in ("EVENT", "NO EVENT")]

    # Diagnostica dei dati originali
    print("=== DIAGNOSTICA DATI ORIGINALI ===")
    original_labels = [d[2] for d in filtered_data]
    original_counter = Counter(original_labels)
    print(f"Distribuzione classi totale: {dict(original_counter)}")

    original_sets = [d[3] for d in filtered_data]
    sets_counter = Counter(original_sets)
    print(f"Distribuzione vincoli set: {dict(sets_counter)}")

    # Analisi per set e classe
    for set_constraint in sets_counter.keys():
        subset = [d for d in filtered_data if d[3] == set_constraint]
        subset_labels = [d[2] for d in subset]
        subset_counter = Counter(subset_labels)
        print(f"Set '{set_constraint}': {dict(subset_counter)}")

    # Suddividi i dati in base al vincolo di set
    trainval_data = [d for d in filtered_data if d[3]
                     in ("train/val", "train/val/test")]
    test_data = [d for d in filtered_data if d[3] == "train/val/test"]

    print("\n=== DOPO SPLIT INIZIALE ===")
    print(f"Train/Val samples: {len(trainval_data)}")
    print(f"Test samples: {len(test_data)}")

    # Verifica test set
    if test_data:
        test_labels = [d[2] for d in test_data]
        test_counter = Counter(test_labels)
        print(f"Test set distribution: {dict(test_counter)}")

        # CONTROLLO CRITICO: verifica che entrambe le classi siano presenti
        if len(test_counter) < 2:
            print("⚠️  ERRORE: Il test set non contiene entrambe le classi!")
            print("⚠️  Questo indica un problema nella assegnazione dei vincoli di set.")
            print(
                "⚠️  Suggerimento: rivedere come vengono assegnati i valori 'train/val/test'")
            missing_class = "NO EVENT" if "EVENT" in test_counter else "EVENT"
            print(f"⚠️  Classe mancante nel test set: {missing_class}")

            # Ritorna comunque i dati, ma avvisa dell'errore
            return [], [], test_data
    else:
        print("⚠️  ERRORE: Test set vuoto!")
        return [], [], []

    # Conta le classi per train/val
    labels_trainval = [d[2] for d in trainval_data]
    counter_trainval = Counter(labels_trainval)
    print(f"Train/Val distribution: {dict(counter_trainval)}")

    if len(counter_trainval) < 2:
        print("⚠️  ERRORE: Train/Val set non contiene entrambe le classi!")
        return trainval_data, [], test_data

    min_class_trainval = min(counter_trainval.values())
    print(f"Min class count for balancing train/val: {min_class_trainval}")

    # Bilancia il dataset train/val 50-50 tra EVENT e NO EVENT
    event_data_trainval = [d for d in trainval_data if d[2] == "EVENT"]
    no_event_data_trainval = [d for d in trainval_data if d[2] == "NO EVENT"]

    np.random.seed(42)
    # Use random.choices or numpy indexing instead of np.random.choice on structured data
    event_indices = np.random.choice(
        len(event_data_trainval), min_class_trainval, replace=False)
    no_event_indices = np.random.choice(
        len(no_event_data_trainval), min_class_trainval, replace=False)

    event_data_trainval = [event_data_trainval[i] for i in event_indices]
    no_event_data_trainval = [no_event_data_trainval[i]
                              for i in no_event_indices]

    balanced_trainval_data = event_data_trainval + no_event_data_trainval
    np.random.shuffle(balanced_trainval_data)

    # Split train/val 82.35%-17.65% (per ottenere 70-15 su totale)
    train_size = int(0.8235 * len(balanced_trainval_data))
    train_data = balanced_trainval_data[:train_size]
    val_data = balanced_trainval_data[train_size:]

    print(f"\n=== RISULTATI FINALI ===")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Bilancia anche il test set solo se entrambe le classi sono presenti
    labels_test = [d[2] for d in test_data]
    counter_test = Counter(labels_test)
    min_class_test = min(counter_test.values()) if len(
        counter_test) == 2 else 0

    if min_class_test > 0:
        event_data_test = [d for d in test_data if d[2] == "EVENT"]
        no_event_data_test = [d for d in test_data if d[2] == "NO EVENT"]

        print(f"Balancing test set - min class: {min_class_test}")

        event_indices_test = np.random.choice(
            len(event_data_test), min_class_test, replace=False)
        no_event_indices_test = np.random.choice(
            len(no_event_data_test), min_class_test, replace=False)

        event_data_test = [event_data_test[i] for i in event_indices_test]
        no_event_data_test = [no_event_data_test[i]
                              for i in no_event_indices_test]

        balanced_test_data = event_data_test + no_event_data_test
        np.random.shuffle(balanced_test_data)
        test_data = balanced_test_data

        print(f"Final test samples after balancing: {len(test_data)}")
    else:
        print("⚠️  Mantengo test set non bilanciato a causa della mancanza di una classe")

    return train_data, val_data, test_data


def assign_set_constraints_stratified(data, test_ratio=0.15, val_ratio=0.15):
    """
    Assegna i vincoli di set in modo stratificato mantenendo le proporzioni delle classi.

    Args:
        data: lista di tuple (feature, ..., label, ...)
        test_ratio: proporzione per il test set
        val_ratio: proporzione per il validation set (sul totale)

    Returns:
        data with updated set constraints
    """
    from collections import defaultdict
    import numpy as np

    # Raggruppa per classe
    class_data = defaultdict(list)
    for item in data:
        label = item[2]  # assumendo che il label sia in posizione 2
        class_data[label].append(item)

    stratified_data = []
    np.random.seed(42)

    for label, items in class_data.items():
        items = list(items)  # copia per non modificare l'originale
        np.random.shuffle(items)

        n_total = len(items)
        n_test = int(test_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_train = n_total - n_test - n_val

        print(
            f"Classe {label}: {n_total} totali -> {n_train} train, {n_val} val, {n_test} test")

        # Assegna i vincoli
        for i, item in enumerate(items):
            # Crea una nuova tupla con il vincolo aggiornato
            if i < n_train:
                constraint = "train/val"
            elif i < n_train + n_val:
                constraint = "train/val"  # val sarà separato dopo nel tuo split_data
            else:
                constraint = "train/val/test"  # questi andranno nel test

            # Aggiorna la tupla (assumendo che il constraint sia in posizione 3)
            updated_item = list(item)
            updated_item[3] = constraint
            stratified_data.append(tuple(updated_item))

    return stratified_data


def fix_existing_data_constraints(data):
    """
    Fix per i tuoi dati esistenti - riassegna i vincoli in modo stratificato
    """
    from collections import defaultdict
    import numpy as np

    # Filtra solo i dati validi
    valid_data = [d for d in data if d[2] in ("EVENT", "NO EVENT")]

    # Raggruppa per classe
    class_data = defaultdict(list)
    for item in valid_data:
        label = item[2]
        class_data[label].append(item)

    print("=== FIXING SET CONSTRAINTS ===")

    fixed_data = []
    np.random.seed(42)

    # Calcola le proporzioni target
    total_items = len(valid_data)
    test_size = int(0.15 * total_items)  # 15% per test

    for label, items in class_data.items():
        items = list(items)
        np.random.shuffle(items)

        n_total = len(items)
        # Mantieni la proporzione di test per classe
        n_test = int(0.15 * n_total)
        n_trainval = n_total - n_test

        print(
            f"Classe {label}: {n_total} totali -> {n_trainval} train/val, {n_test} test")

        # Assegna i nuovi vincoli
        for i, item in enumerate(items):
            updated_item = list(item)
            if i < n_trainval:
                updated_item[3] = "train/val"
            else:
                updated_item[3] = "train/val/test"

            fixed_data.append(tuple(updated_item))

    # Verifica il risultato
    print("\n=== VERIFICA DOPO FIX ===")
    from collections import Counter

    labels = [d[2] for d in fixed_data]
    constraints = [d[3] for d in fixed_data]

    print(f"Distribuzione classi: {dict(Counter(labels))}")
    print(f"Distribuzione vincoli: {dict(Counter(constraints))}")

    # Verifica stratificazione
    for constraint in set(constraints):
        subset = [d for d in fixed_data if d[3] == constraint]
        subset_labels = [d[2] for d in subset]
        subset_counter = Counter(subset_labels)
        print(f"Set '{constraint}': {dict(subset_counter)}")

    return fixed_data


def unpack_data(data):
    if len(data) == 0:
        return [], [], [], [], [], []
    return [list(x) for x in zip(*data)]

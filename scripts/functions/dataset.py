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

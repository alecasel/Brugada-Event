
from matplotlib import pyplot as plt
import numpy as np


def extract_sequences(sequences_vector):
    sequences = []
    for cell in np.squeeze(sequences_vector):
        sequences.append(np.squeeze(cell))
    return np.vstack(sequences)


def convert_from_mat_format(data,
                            patients_list,
                            ecgs_list,
                            labels_event_list,
                            sets_list,
                            leads_list,
                            signals_list):

    # Extract data attributes
    keys = list(data.keys())
    # Remove the hidden keys
    keys = [key for key in keys if not key.startswith('__')]

    for k in keys:

        if k == 'Z':  # ECG Path
            ecg_paths = data[k].flatten()
            for ep in ecg_paths:
                ecg_path = str(ep[0])
                if "Holter" not in ecg_path:
                    patient = ecg_path.split('\\')[-2]
                else:
                    patient = ecg_path.split('\\')[-3]
                ecg_id = ecg_path.split('\\')[-1].replace('.xml', '')
                if "EVENTO" in ecg_path:
                    label_event = "EVENT"
                    if "_Assenza Storia Clinica" in ecg_path:
                        set_data = "test"
                    else:
                        set_data = "train/val"
                elif "ASINTOMATICI" in ecg_path:
                    label_event = "NO EVENT"
                    set_data = "train/val/test"

                # Update the lists
                patients_list.append(patient)
                ecgs_list.append(ecg_id)
                labels_event_list.append(label_event)
                sets_list.append(set_data)

        elif k == 'L':  # Lead
            leads = data[k].flatten()
            for ld in leads:
                lead = str(ld[0])
                # Update the leads list
                leads_list.append(lead)

        elif k == 'X':  # Signal
            signals_list.extend(extract_sequences(data[k]))

    return patients_list, ecgs_list, labels_event_list, \
        sets_list, leads_list, signals_list


def plot_heartbeat_on_ecg(signal,
                          lead,
                          patient,
                          gt_label,
                          idx,
                          tot_idx):
    """
    Plots a single heartbeat on an ECG plot.
    """

    _, ax = plt.subplots(figsize=(6, 6))
    time_axis = np.arange(len(signal))
    ax.plot(time_axis, signal, color='blue',
            linewidth=0.7, label='Original Signal')

    ax.set_title(
        f"[{idx+1}/{tot_idx}] " +
        f"Patient={patient}, Lead={lead}, Class={gt_label}",
        fontsize=10)
    ax.set_xlabel("Time (ms)", fontsize=6)
    ax.set_ylabel("Amplitude (mV)", fontsize=6)
    ax.set_xlim(0, 850)
    ax.set_ylim(-1.125, 1.125)

    major_xticks = np.arange(0, 851, 200)
    minor_xticks = np.arange(0, 851, 40)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xticklabels(major_xticks, fontsize=8)

    major_yticks = np.arange(-1.0, 1.1, 0.5)
    minor_yticks = np.arange(-1.0, 1.1, 0.1)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_yticklabels(major_yticks, fontsize=8)

    ax.grid(which='minor', color='pink', linestyle='-', linewidth=0.5)
    ax.grid(which='major', color='pink', linestyle='-', linewidth=1.5)
    ax.legend(fontsize=8)

    plt.show()

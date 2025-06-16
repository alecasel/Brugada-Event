import os
from matplotlib import pyplot as plt
import numpy as np


def _get_test_data(filtered_data,
                   frequency,
                   index_S):
    """
    Prepara i dati di test per la predizione della Sindrome di Brugada
    estraendo finestre attorno ai picchi S e confrontando le finestre vicine.
    """

    from scripts.functions.ecg_extraction import \
        longest_common_consecutive_match

    left_l = int((frequency * 300) / 1000)
    right_l = int((frequency * 551) / 1000)

    X_data_test = []
    index_S_cleaned = []

    for j in range(len(index_S)):
        if (index_S[j] - left_l > 0 and
                index_S[j] + right_l < len(filtered_data)):
            signal = filtered_data[index_S[j] - left_l:index_S[j] + right_l]

            if len(signal) < 851:
                padding = 851 - len(signal)
                signal = np.pad(signal, (0, padding), mode='edge')
            elif len(signal) > 851:
                signal = signal[:851]

            X_data_test.append(signal)
            index_S_cleaned.append(index_S[j])

    # Confronto intelligente solo tra finestre vicine
    to_remove = set()
    for i in range(len(X_data_test)):
        for j in range(i + 1, len(X_data_test)):
            if i in to_remove or j in to_remove:
                continue

            if abs(index_S_cleaned[i] - index_S_cleaned[j]) > 500:
                continue

            sig1 = X_data_test[i].flatten()
            sig2 = X_data_test[j].flatten()

            max_common, start_i, start_j = longest_common_consecutive_match(
                sig1, sig2, 1e-3)
            similarity_ratio = max_common / 851

            if similarity_ratio >= 0.5:
                if sig1.min() < sig2.min():
                    to_remove.add(j)
                else:
                    to_remove.add(i)

            elif 0.4 <= similarity_ratio < 0.5:
                print(
                    f"\n⚠️ Somiglianza borderline "
                    f"({similarity_ratio*100:.2f}%) tra finestra {i} e {j}.")

                sig1_min = sig1.min()
                sig2_min = sig2.min()

                # Controlla se i due picchi sono molto simili
                if abs(sig1_min - sig2_min) < 0.1:
                    print(
                        "⚠️ I due picchi sono molto simili. Entrambe " +
                        "le finestre saranno conservate automaticamente.")
                    continue

                # Eliminazione picchi troppo bassi irreali
                if filtered_data[index_S_cleaned[i]] >= -0.05 \
                        and filtered_data[index_S_cleaned[i]] <= 0.05:
                    to_remove.add(i)
                    continue
                elif filtered_data[index_S_cleaned[j]] >= -0.05 \
                        and filtered_data[index_S_cleaned[j]] <= 0.05:
                    to_remove.add(j)
                    continue

                # fig, axs = plt.subplots(2, 1, figsize=(
                #     12, 6), sharex=True, sharey=True)

                # axs[0].plot(sig1, label=f'Finestra {i}', color='blue')
                # axs[0].axvspan(start_i, start_i + max_common,
                #                color='red', alpha=0.2)
                # axs[0].set_title(f'Finestra {i} - Min S: {sig1_min:.3f}')
                # axs[0].legend()
                # axs[0].grid(True)

                # axs[1].plot(sig2, label=f'Finestra {j}', color='green')
                # axs[1].axvspan(start_j, start_j + max_common,
                #                color='red', alpha=0.2)
                # axs[1].set_title(f'Finestra {j} - Min S: {sig2_min:.3f}')
                # axs[1].legend()
                # axs[1].grid(True)

                # plt.tight_layout()
                # plt.show()

                # user_input = input(
                #     "Quale finestra vuoi MANTENERE? " +
                #     "(sopra/sotto/entrambe/scarta): ").strip().lower()
                # if user_input == 'sopra':
                #     to_remove.add(j)
                # elif user_input == 'sotto':
                #     to_remove.add(i)
                # elif user_input == 'scarta':
                #     to_remove.add(i)
                #     to_remove.add(j)

    X_data_test_final = [x for k, x in enumerate(
        X_data_test) if k not in to_remove]
    X_data_test_final = np.expand_dims(np.array(X_data_test_final), axis=-1)

    return X_data_test_final


def _make_prediction(actual_filtered_data_lead,
                     frequency,
                     model,
                     consider_embeddings=False):
    """
    Effettua la predizione della Sindrome di Brugada
    utilizzando un modello pre-addestrato.
    """

    from scripts.functions.ecg_extraction import S_peaks_V1

    index_S = S_peaks_V1(
        actual_filtered_data_lead, 1/frequency)

    X_data_test = _get_test_data(
        actual_filtered_data_lead, frequency, index_S)

    # Reshape dei dati per adattarli al modello
    X_test_reshaped = X_data_test.reshape(X_data_test.shape[0],
                                          X_data_test.shape[1],
                                          1)

    # Predizioni
    Y_pred = model.predict(X_test_reshaped, verbose=0)

    # If this is the student model, return only the predictions
    if isinstance(Y_pred, tuple):
        if not consider_embeddings:
            return Y_pred[0]
        else:
            return Y_pred[1]

    return Y_pred


def process_folder(xml_folder,
                   output_file,
                   model,
                   consider_embeddings=False,
                   lead_for_umap=['V1'],
                   num_patients=10):
    """
    Processa tutti i file XML in una cartella specificata,
    estrae i dati ECG, effettua le predizioni e salva i risultati
    in un file Excel.
    """

    from scripts.functions.ecg_extraction import import_ecg_data, lowpass
    from scripts.functions.utils import save_predictions_to_excel

    if os.path.exists(output_file):
        user_input = input(
            f"The file {output_file} already exists. " +
            "Do you want to delete (D) it or overwrite (O) it? " +
            "(delete/overwrite): ").strip().lower()
        if user_input == 'd':
            os.remove(output_file)
        elif user_input == 'o':
            raise ValueError("Invalid input. " +
                             "Please enter 'd' or 'o'.")

    if consider_embeddings:
        # Inizializza le liste per salvare embeddings e nomi pazienti
        all_embeddings = []
        all_patients = []
        all_classes = []

    # Processa solo le dirette sottocartelle
    for folder in sorted(os.listdir(xml_folder)):
        if not os.path.isdir(os.path.join(xml_folder, folder)):
            continue
        if folder == "SINCOPI" or folder == "SINCOPI VAGALI":
            continue
        count_patients = 0
        folder_path = os.path.join(xml_folder, folder)
        subfolders = sorted(os.listdir(folder_path))
        for item in subfolders:
            if "_Assenza" in item:
                continue
            if count_patients == num_patients:
                continue
            item_path = os.path.join(xml_folder, folder, item)
            if os.path.isdir(item_path):  # È una directory
                subfolder_name = item  # Nome della sottocartella
                if any(xml_file.endswith(".xml") for xml_file in os.listdir(item_path)):
                    count_patients += 1
                for xml_file in sorted(os.listdir(item_path)):
                    if xml_file.endswith(".xml"):
                        xml_file_path = os.path.join(item_path, xml_file)
                        print(xml_file_path)
                        ecg_data, _, frequency = import_ecg_data(xml_file_path)
                        ecg_data_filtered = {}

                        if not consider_embeddings:
                            leads = ['V1', 'V2']
                        else:
                            leads = lead_for_umap

                        # Filtra i dati ECG con un filtro passa-basso
                        for lead in leads:
                            if lead in ecg_data:
                                ecg_data_filtered[lead] = lowpass(
                                    ecg_data[lead]["V"], cutoff=10,
                                    fs=frequency)

                                if np.allclose(
                                        ecg_data_filtered[lead], 0, atol=1e-6):
                                    print(
                                        f"⚠️ {lead} contiene tutti zero. Skip")
                                    continue

                                predictions = _make_prediction(
                                    ecg_data_filtered[lead], frequency, model,
                                    consider_embeddings=consider_embeddings)
                                if predictions is not None:
                                    if not consider_embeddings:
                                        save_predictions_to_excel(lead,
                                                                  predictions,
                                                                  xml_file,
                                                                  output_file)
                                    else:
                                        # Salva embeddings e nome sottocartella
                                        all_embeddings.append(predictions[1])
                                        all_patients.append(subfolder_name)
                                        all_classes.append(folder)

    if consider_embeddings:
        return all_embeddings, all_patients, all_classes


def _get_umap_patient_clusters(Z,
                               umap_n_neighbors=15,
                               umap_min_dist=0.1,
                               umap_metric='euclidean'):
    """
    Calcola le coordinate UMAP per i pazienti
    e restituisce le coordinate UMAP dei pazienti
    che hanno almeno un campione con lead_mask True.
    """

    import umap

    reducer = umap.UMAP(n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        metric=umap_metric,
                        random_state=42)

    z_umap = reducer.fit_transform(Z)

    return z_umap


def plot_umap_patient_clusters(Z,
                               patient_ids,
                               classes,
                               n_clusters=2,
                               save_path=None):
    """
    Visualizza pazienti confrontando cluster predetti vs ground truth.

    Args:
        Z: Features dei pazienti
        patient_ids (ndarray): ID dei pazienti (n_samples,)
        classes (ndarray): Ground truth classes (n_samples,)
        n_clusters (int): Numero di cluster da identificare
        save_path (str or None): Percorso file immagine (PNG).

    Returns:
        dict: Risultati con metriche di valutazione
    """

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    z_patients = []
    name_patients = []
    class_patients = []

    for i, z_patient in enumerate(Z):
        for z_single in z_patient:
            z_patients.append(z_single)
            name_patients.append(patient_ids[i])
            class_patients.append(classes[i])

    z_umap = _get_umap_patient_clusters(z_patients)

    unique_patients = np.unique(name_patients)
    centroids = []
    patient_classes = []

    for patient in unique_patients:
        mask = np.array(name_patients) == str(patient)
        centroid = np.mean(z_umap[mask], axis=0)
        centroids.append(centroid)
        # Prendi la classe del paziente (dovrebbe essere uguale per tutti i suoi punti)
        patient_class = np.array(class_patients)[mask][0]
        patient_classes.append(patient_class)

    centroids = np.array(centroids)
    patient_classes = np.array(patient_classes)

    # Clustering sui centroidi dei pazienti
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(centroids)

    # Calcola accuratezza precisa della classificazione dei pazienti
    accuracy, optimal_mapping, correct_mask = _check_patient_classification_accuracy(
        cluster_labels, patient_classes)

    # Calcola anche ARI per confronto
    ari_score = adjusted_rand_score(patient_classes, cluster_labels)
    nmi_score = normalized_mutual_info_score(patient_classes, cluster_labels)

    # Crea mapping
    patient_to_cluster = dict(zip(unique_patients, cluster_labels))
    patient_to_class = dict(zip(unique_patients, patient_classes))

    # Colori per ground truth (forme) e cluster (colori di riempimento)
    unique_classes = np.unique(patient_classes)
    unique_clusters = np.unique(cluster_labels)

    # Crea mapping da classe a indice numerico
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

    class_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    cluster_colors = plt.cm.get_cmap("Set1", n_clusters)

    # Figura con tre subplot per chiarezza
    fig = plt.figure(figsize=(24, 8))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    # SUBPLOT 1: Solo Ground Truth (forme diverse)
    class_colors_gt = plt.cm.get_cmap("Set2", len(unique_classes))
    for patient in unique_patients:
        mask = np.array(name_patients) == str(patient)
        true_class = patient_to_class[patient]
        class_idx = class_to_idx[true_class]

        color = class_colors_gt(class_idx)
        marker = class_markers[class_idx % len(class_markers)]

        # Punti del paziente
        ax1.scatter(z_umap[mask, 0], z_umap[mask, 1],
                    c=[color], marker=marker, alpha=0.8, s=120,
                    edgecolors='black', linewidth=1)

        # Annotazione
        centroid = np.mean(z_umap[mask], axis=0)
        label = patient.split(' ')[0]
        ax1.annotate(f'{true_class}|{label}', centroid,
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, fontweight='bold')

    ax1.set_xlabel("UMAP 1", fontsize=12)
    ax1.set_ylabel("UMAP 2", fontsize=12)
    ax1.set_title('Ground Truth Classes', fontsize=14, fontweight='bold')

    # Legenda ground truth
    gt_legend = []
    for cls in unique_classes:
        class_idx = class_to_idx[cls]
        color = class_colors_gt(class_idx)
        marker = class_markers[class_idx % len(class_markers)]
        gt_legend.append(plt.Line2D([0], [0], marker=marker, color=color,
                                    linestyle='None', markersize=10,
                                    label=f'{cls}'))
    ax1.legend(handles=gt_legend, title='True Classes', loc='best')

    # SUBPLOT 2: Solo Cluster Predetti (colori diversi, forme uguali)
    for patient in unique_patients:
        mask = np.array(name_patients) == str(patient)
        cluster = patient_to_cluster[patient]

        color = cluster_colors(cluster)

        # Punti del paziente (tutti cerchi)
        ax2.scatter(z_umap[mask, 0], z_umap[mask, 1],
                    c=[color], marker='o', alpha=0.8, s=30,
                    edgecolors='black', linewidth=0.5)

        # Annotazione
        centroid = np.mean(z_umap[mask], axis=0)
        label = patient.split(' ')[0]
        ax2.annotate(f'{label}', centroid,
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, fontweight='bold')

        # Aggiungi una croce vicino al centroide con il colore del cluster
        ax2.scatter(centroid[0], centroid[1],
                    c=cluster_colors(cluster), marker='x', s=100,
                    edgecolors='black', linewidth=1.5)

    ax2.set_xlabel("UMAP 1", fontsize=12)
    ax2.set_ylabel("UMAP 2", fontsize=12)
    ax2.set_title('Predicted Clusters', fontsize=14, fontweight='bold')

    # Legenda cluster
    cluster_legend = []
    for cluster in unique_clusters:
        color = cluster_colors(cluster)
        cluster_legend.append(plt.Line2D([0], [0], marker='o', color=color,
                                         linestyle='None', markersize=10,
                                         label=f'Cluster {cluster}'))
    ax2.legend(handles=cluster_legend, title='Predicted Clusters', loc='best')

    # SUBPLOT 3: Patient Classification Accuracy (verde=corretto, rosso=sbagliato)
    for i, patient in enumerate(unique_patients):
        mask = np.array(name_patients) == str(patient)
        cluster = cluster_labels[i]
        true_class = patient_classes[i]
        centroid = centroids[i]

        # Verde se classificato correttamente, rosso se sbagliato
        is_correct = correct_mask[i]
        color = 'green' if is_correct else 'red'

        # Punti del paziente
        ax3.scatter(z_umap[mask, 0], z_umap[mask, 1],
                    c=color, alpha=0.6, s=80)

        # Centroide più grande e marcato
        ax3.scatter(centroid[0], centroid[1],
                    c=color, s=200, marker='X',
                    edgecolors='black', linewidth=2)

        # Annotazione con info cluster e classe
        label = patient.split(' ')[0]
        predicted_class = optimal_mapping.get(cluster, '?')
        ax3.annotate(f'{label}\nC{cluster}→{predicted_class}|T{true_class}', centroid,
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, fontweight='bold')

    ax3.set_xlabel("UMAP 1", fontsize=12)
    ax3.set_ylabel("UMAP 2", fontsize=12)
    ax3.set_title(f'Patient Classification Accuracy: {accuracy:.1%}\n({np.sum(correct_mask)}/{len(correct_mask)} patients correct)',
                  fontsize=14, fontweight='bold')

    # Legenda
    accuracy_legend = [
        plt.Line2D([0], [0], marker='X', color='green', linestyle='None',
                   markersize=12, label=f'Correct ({np.sum(correct_mask)} patients)'),
        plt.Line2D([0], [0], marker='X', color='red', linestyle='None',
                   markersize=12, label=f'Wrong ({len(correct_mask)-np.sum(correct_mask)} patients)')
    ]
    ax3.legend(handles=accuracy_legend,
               title='Patient Classification', loc='best')

    # Aggiungi info sul mapping ottimale
    mapping_text = "Optimal mapping:\n" + \
        "\n".join([f"Cluster {k} → Class {v}" for k,
                  v in optimal_mapping.items()])
    ax3.text(0.02, 0.98, mapping_text, transform=ax3.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

    # Salvataggio opzionale
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return {
        'patient_to_cluster': dict(zip(unique_patients, cluster_labels)),
        'patient_to_class': dict(zip(unique_patients, patient_classes)),
        'accuracy': accuracy,
        'optimal_mapping': optimal_mapping,
        'correct_patients': unique_patients[correct_mask],
        'wrong_patients': unique_patients[~correct_mask],
        'ari_score': ari_score,
        'nmi_score': nmi_score,
        'centroids': centroids,
        'cluster_labels': cluster_labels,
        'patient_classes': patient_classes
    }


def _check_patient_classification_accuracy(cluster_labels, true_classes):
    """
    Calcola l'accuratezza della classificazione
    dei pazienti basata sui centroidi.
    Trova il miglior mapping tra cluster e classi e
    conta i pazienti classificati correttamente.

    Returns:
        accuracy: percentuale di pazienti classificati correttamente
        mapping: dizionario cluster -> classe ottimale
        correct_mask: array booleano indicante quali pazienti
        sono classificati bene
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    # Crea matrice di confusione
    unique_classes = np.unique(true_classes)

    cm = confusion_matrix(true_classes, cluster_labels,
                          labels=unique_classes)

    # Trova il miglior mapping cluster->classe usando l'algoritmo ungherese
    # Dobbiamo massimizzare, quindi usiamo -cm
    row_indices, col_indices = linear_sum_assignment(-cm)

    # Crea mapping ottimale
    optimal_mapping = {}
    for i, j in zip(row_indices, col_indices):
        optimal_mapping[j] = unique_classes[i]  # cluster j -> classe i

    # Calcola accuratezza con mapping ottimale
    correct_predictions = 0
    correct_mask = np.zeros(len(cluster_labels), dtype=bool)

    for i, (cluster, true_class) in enumerate(zip(cluster_labels, true_classes)):
        predicted_class = optimal_mapping.get(cluster, -1)
        if predicted_class == true_class:
            correct_predictions += 1
            correct_mask[i] = True

    accuracy = correct_predictions / len(cluster_labels)

    return accuracy, optimal_mapping, correct_mask

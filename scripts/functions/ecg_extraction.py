import math
import numpy as np


def import_ecg_data(xml_file):
    """
    Importa i dati ECG da un file XML.

    :param file_path: Percorso del file XML contenente i dati ECG.
    :return: Dizionario contenente i dati ECG estratti.
    """

    import xml.etree.ElementTree as ET

    # Carica il file XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    namespaces = {'ns': 'urn:ge:sapphire:dcar_1'}

    # Trova e filtra le ecgWaveform con asizeVT = 10000
    waveforms = [wf for wf in root.findall(
        ".//ns:ecgWaveformMXG/ns:ecgWaveform", namespaces=namespaces)
        if int(wf.get("asizeVT", "0")) == 10000]

    # Estrai i dati delle onde ECG
    ecg_data = {}
    for waveform in waveforms:
        lead = waveform.get("lead")
        asizeVT = int(waveform.get("asizeVT", "0"))
        vt_data = waveform.get("V")
        if lead and vt_data:
            ecg_data[lead] = {
                "asizeVT": asizeVT,
                # Conversione come in MATLAB
                "V": [4.88 * int(value) / 1000 for value in vt_data.split()]
            }

    # Estrai frequenza di campionamento
    sample_rate = root.find(".//ns:sampleRate", namespaces=namespaces)
    if sample_rate is not None:
        # Valore di default: 1000 Hz
        frequency = int(sample_rate.get("V", "1000"))

    # Calcola il tempo
    if "V1" in ecg_data:
        time = [i / frequency for i in range(ecg_data["V1"]["asizeVT"])]

    return ecg_data, time, frequency


def waveft(wavelet,
           omega,
           scales):
    """
    waveft(wavelet, omega, scales)

    Computes the Fourier transform of a wavelet at certain scales.

    Parameters
    ----------
    wavelet : string
        Name of wavelet
    omega : array_like
        Array containing frequency values in Hz
        at which the transform is evaluated.
    scales : array_like
        Vector containing the scales used for the wavelet analysis.

    Returns
    -------
    wft : array_like
        (num_scales x num_freq) Array containing the wavelet Fourier transform
    freq : array_like
        Array containing frequency values
    """
    wname = wavelet
    num_freq = omega.size
    num_scales = scales.size
    wft = np.zeros([num_scales, num_freq])

    if wname == 'morl':
        gC = 6
        mul = 2
        for jj, scale in enumerate(scales):
            expnt = -(scale * omega - gC) ** 2 / 2 * (omega > 0)
            wft[jj, ] = mul * np.exp(expnt) * (omega > 0)

        fourier_factor = gC / (2 * np.pi)
        frequencies = fourier_factor / scales

    else:
        raise Exception

    return wft, frequencies


def getDefaultScales(wavelet,
                     n,
                     ds):
    """
    getDefaultScales(wavelet, n, ds)

    Calculate default scales given a wavelet and a signal length.

    Parameters
    ----------
    wavelet : string
        Name of wavelet
    n : int
        Number of samples in a given signal
    ds : float
        Scale resolution (inverse of number of voices in octave)

    Returns
    -------
    s0 : int
        Smallest useful scale
    ds : float
        Scale resolution (inverse of number of voices in octave).
        Here for legacy reasons; implementing more wavelets
        will need this output.
    scales : array_like
        Array containing default scales.
    """
    wname = wavelet
    nv = 1 / ds

    if wname == 'morl':

        # Smallest useful scale (default 2 for Morlet)
        s0 = 2

        # Determine longest useful scale for wavelet
        max_scale = n // (np.sqrt(2) * s0)
        if max_scale <= 1:
            max_scale = n // 2
        max_scale = np.floor(nv * np.log2(max_scale))
        a0 = 2 ** ds
        scales = s0 * a0 ** np.arange(0, max_scale + 1)
    else:
        raise Exception

    return s0, ds, scales


def cwt(data,
        wavelet_name,
        sampling_frequency=1.):
    """
    cwt(data, scales, wavelet)

    One dimensional Continuous Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    wavelet_name : Wavelet object or name
        Wavelet to use. Currently, only the Morlet wavelet
        is supported ('morl').
    sampling_frequency : float
        Sampling frequency for frequencies output (optional)

    Returns
    -------
    coefs : array_like
        Continous wavelet transform of the input signal for the given scales
        and wavelet
    frequencies : array_like
        if the unit of sampling period are seconds and given, than frequencies
        are in hertz. Otherwise Sampling period of 1 is assumed.

    Notes
    -----
    Size of coefficients arrays is automatically calculated
    given the wavelet and the data length. Currently, only the
    Morlet wavelet is supported.

    Examples
    --------
    fs = 1e3
    t = np.linspace(0, 1, fs+1, endpoint=True)
    x = np.cos(2*np.pi*32*t) * np.logical_and(t >= 0.1, t < 0.3) +
        + np.sin(2*np.pi*64*t) * (t > 0.7)
    wgnNoise = 0.05 * np.random.standard_normal(t.shape)
    x += wgnNoise
    c, f = cwt.cwt(x, 'morl', sampling_frequency=fs, plot_scalogram=True)
    """

    # Currently only supported for Morlet wavelets
    if wavelet_name == 'morl':
        data -= np.mean(data)
        n_orig = data.size
        nv = 10
        ds = 1 / nv
        fs = sampling_frequency

        # Pad data symmetrically
        padvalue = n_orig // 2
        x = np.concatenate(
            (np.flipud(data[0:padvalue]), data, np.flipud(data[-padvalue:])))
        n = x.size

        # Define scales
        _, _, wavscales = getDefaultScales(wavelet_name, n_orig, ds)
        num_scales = wavscales.size

        # Frequency vector sampling the Fourier transform of the wavelet
        omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
        omega *= (2 * np.pi) / n
        omega = np.concatenate((np.array([0]),
                                omega,
                                -omega[np.arange(math.floor((n - 1) / 2),
                                                 0, -1, dtype=int) - 1]))

        # Compute FFT of the (padded) time series
        f = np.fft.fft(x)

        # Loop through all the scales and compute wavelet Fourier transform
        psift, freq = waveft(wavelet_name, omega, wavscales)

        # Inverse transform to obtain the wavelet coefficients.
        cwtcfs = np.fft.ifft(np.kron(np.ones([num_scales, 1]), f) * psift)
        cfs = cwtcfs[:, padvalue:padvalue + n_orig]
        freq = freq * fs

        return cfs, freq
    else:
        raise Exception


def S_peaks_V1(V1,
               dt):
    """
    Ricerca dei picchi S tramite Continuous Wavelet Transform
    e iterazione per trovare i minimi in V1 vicini a ciascun picco
    della trasformata, con filtraggio dinamico dei picchi in base
    all'ampiezza del segnale ricostruito.

    Se il numero di picchi filtrati (con soglia iniziale al 50% del massimo)
    è inferiore a 8, la soglia viene abbassata progressivamente (diminuendo
    il fattore di soglia) fino a raggiungere almeno 8 picchi
    o un limite minimo.

    Args:
        V1 (numpy.ndarray): Segnale ECG (derivazione V1).
        dt (float): Passo temporale (in secondi).

    Returns:
        numpy.ndarray: Indici (nel vettore V1) dei picchi S rilevati.
    """

    from scipy.signal import find_peaks

    # Parametri
    index = []
    t = 10  # numero di iterazioni per affinare la ricerca dei picchi S
    # finestra di 100 ms (circa 50 campioni a 1kHz)
    toll = int(np.floor(50 / (dt * 1000)))

    # Calcolo della Continuous Wavelet Transform
    coefficients, _ = cwt(V1, 'morl', sampling_frequency=1/dt)

    # Ricostruzione (approssimata) del segnale
    # come somma dei moduli delle wavelet
    scales = np.arange(1, coefficients.shape[0] + 1)
    reconstructed = np.sum(np.abs(coefficients) /
                           np.sqrt(scales[:, np.newaxis]), axis=0)

    # Calcola la lunghezza effettiva di V1
    # escludendo la coda di valori tutti uguali (con alta precisione)
    effective_length = len(V1)

    # Calcola la durata in secondi
    duration_sec = effective_length * dt

    # Frequenza minima attesa (in bpm)
    min_hr_bpm = 45

    # Calcola il numero minimo di picchi attesi
    min_num_peaks = int(np.floor((min_hr_bpm / 60) * duration_sec))

    # 1) Trova i picchi nel segnale ricostruito
    peaks, _ = find_peaks(reconstructed)

    # Controlla se c'è un picco all'inizio
    if len(peaks) > 0:  # Se esistono picchi rilevati
        # Controlliamo se x[0] < 300
        if peaks[0] < 300:
            peaks = peaks[1:]  # Rimuoviamo il primo picco

        # Valore del picco massimo trovato
        max_peak_value = reconstructed[peaks].max()

    else:
        return np.array([])  # Nessun picco trovato

    # 2) Applica una soglia dinamica
    threshold_factor = 0.8
    dynamic_threshold = threshold_factor * max_peak_value
    peaks_filtered = [
        p for p in peaks if reconstructed[p] >= dynamic_threshold]

    # Se il numero di picchi filtrati è inferiore al minimo atteso,
    # abbassa progressivamente la soglia
    while len(peaks_filtered) < min_num_peaks and threshold_factor > 0.15:
        threshold_factor -= 0.05
        dynamic_threshold = threshold_factor * max_peak_value
        peaks_filtered = [
            p for p in peaks if reconstructed[p] >= dynamic_threshold]

    # Remove duplicates
    min_distance = 200
    index_sorted = np.sort(peaks_filtered)
    filtered_index = []
    for idx_val in index_sorted:
        if len(filtered_index) == 0:
            filtered_index.append(idx_val)
        else:
            if (idx_val - filtered_index[-1]) >= min_distance:
                filtered_index.append(idx_val)

    peaks_filtered_2 = np.array(filtered_index)

    # 3) Associa i picchi del segnale ricostruito
    # ai minimi locali in V1 vicini (entro ± toll)
    for peak in peaks_filtered_2:
        if peak - toll > 0 and peak + toll < len(V1):
            local_min_idx = np.argmin(
                V1[peak - toll:peak + toll]) + (peak - toll)
            index.append(local_min_idx)
        elif peak - toll < 0:
            local_min_idx = np.argmin(V1[:peak + toll])
            index.append(local_min_idx)
        else:
            local_min_idx = np.argmin(V1[peak - toll:]) + (peak - toll)
            index.append(local_min_idx)

    # 4) Iterazione per affinare i risultati:
    # ricerca locale ripetuta per "centrare" meglio i picchi S
    for _ in range(t):
        index_new = []
        for idx in index:
            if idx - toll > 0 and idx + toll < len(V1):
                ind = np.argmin(V1[idx - toll:idx + toll]) + (idx - toll)
                index_new.append(ind)
            elif idx - toll < 0:
                ind = np.argmin(V1[:idx + 2 * toll])
                index_new.append(ind)
            elif idx + toll >= len(V1):
                ind = np.argmin(V1[idx - 2 * toll:]) + idx - 2 * toll
                index_new.append(ind)
        index_final = np.unique(index_new)

    return np.array(index_final)


def longest_common_consecutive_match(sig1,
                                     sig2,
                                     tolerance):
    """
    Trova la lunghezza della corrispondenza consecutiva più lunga
    tra due segnali.
    Utilizza un algoritmo di confronto veloce per identificare la
    corrispondenza più lunga di segmenti consecutivi che sono simili entro
    una certa tolleranza.
    """
    max_len = 0
    start_i = 0
    start_j = 0
    len1, len2 = len(sig1), len(sig2)

    for i in range(len1):
        for j in range(len2):
            k = 0
            while (
                i + k < len1 and
                j + k < len2 and
                abs(sig1[i + k] - sig2[j + k]) < tolerance
            ):
                k += 1
            if k > max_len:
                max_len = k
                start_i = i
                start_j = j

    return max_len, start_i, start_j


def lowpass(signal,
            cutoff,
            fs,
            order=6):
    """
    Replica la funzione lowpass di MATLAB.
    :param signal: Segnale da filtrare
    :param cutoff: Frequenza di taglio in Hz
    :param fs: Frequenza di campionamento in Hz
    :param order: Ordine del filtro
    :return: Segnale filtrato
    """

    from scipy.signal import butter, filtfilt

    def _design_filter(cutoff, fs, order=6):
        nyquist = fs / 8
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='low')
        return b, a

    def _apply_filter(b, a, signal):
        return filtfilt(b, a, signal, method='pad', padtype='odd')

    b, a = _design_filter(cutoff, fs, order)
    return _apply_filter(b, a, signal)

import os
from pathlib import Path

import mne
import numpy as np
import pywt
from tqdm import tqdm


# ============================================================
# KONFIGURACJA
# ============================================================

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

RAW_DATA_DIR = str(PROJECT_DIR / "data" / "raw")
OUTPUT_DIR = str(PROJECT_DIR / "data" / "spectrograms")

# długość epoki po pojawieniu się bodźca
TMIN = 0.0
TMAX = 2.0

# filtracja EEG
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
NOTCH_FREQ = 50.0

# wavelet
WAVELET = "morl"
SCALES = np.arange(1, 64)

# sampling rate docelowy
RESAMPLE_FREQ = 256


# ============================================================
# TWORZENIE FOLDERÓW
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNKCJA: PREPROCESSING EEG
# ============================================================


def preprocess_raw(raw):
    """
    Wykonuje podstawowy preprocessing EEG.
    """

    print("Filtering...")

    # bandpass
    raw.filter(LOW_FREQ, HIGH_FREQ)

    # notch 50 Hz
    raw.notch_filter(NOTCH_FREQ)

    # resampling
    raw.resample(RESAMPLE_FREQ)

    return raw

def create_wavelet_spectrogram(signal):
    """
    Generuje spektrogram waveletowy dla jednego kanału EEG.

    Input:
        signal -> (time,)

    Output:
        power -> (freq, time)
    """

    coefficients, frequencies = pywt.cwt(
        signal,
        SCALES,
        WAVELET
    )

    power = np.abs(coefficients)

    return power.astype(np.float32)

# ============================================================
# FUNKCJA: NORMALIZACJA
# ============================================================


def normalize_spectrogram(spec):
    """
    Standaryzacja spektrogramu.
    """

    mean = np.mean(spec)
    std = np.std(spec)

    if std < 1e-6:
        std = 1e-6

    spec = (spec - mean) / std

    return spec

# ============================================================
# FUNKCJA: GENEROWANIE SPEKTROGRAMÓW DLA EPOKI
# ============================================================


def epoch_to_spectrogram(epoch_data):
    """
    Konwersja pojedynczej epoki EEG.

    Input:
        epoch_data -> (channels, time)

    Output:
        spectrograms -> (channels, freq, time)
    """

    channel_spectrograms = []

    for channel_signal in epoch_data:

        spec = create_wavelet_spectrogram(channel_signal)

        spec = normalize_spectrogram(spec)

        channel_spectrograms.append(spec)

    spectrograms = np.stack(channel_spectrograms)

    return spectrograms.astype(np.float32)

# ============================================================
# FUNKCJA: TWORZENIE EVENTÓW
# ============================================================


def create_dummy_events(raw, every_seconds=3):
    """
    PRZYKŁADOWA funkcja tworzenia eventów.

    Zakłada, że co kilka sekund pojawiał się nowy obraz.

    UWAGA:
    W prawdziwym projekcie powinieneś używać
    prawdziwych timestampów bodźców.
    """

    sfreq = raw.info["sfreq"]

    total_samples = raw.n_times

    step = int(every_seconds * sfreq)

    samples = np.arange(step, total_samples - step, step)

    events = []

    for sample in samples:
        events.append([sample, 0, 1])

    events = np.array(events)

    return events

# ============================================================
# GŁÓWNA FUNKCJA PRZETWARZANIA EDF
# ============================================================


def process_edf(edf_path):

    print("=" * 60)
    print(f"Processing: {edf_path}")
    print("=" * 60)

    # --------------------------------------------------------
    # Wczytanie EDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(edf_path, preload=True)

    print(raw)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    raw = preprocess_raw(raw)

    # --------------------------------------------------------
    # Eventy
    # --------------------------------------------------------

    events = create_dummy_events(raw)

    print(f"Number of events: {len(events)}")

    # --------------------------------------------------------
    # Epoching
    # --------------------------------------------------------

    epochs = mne.Epochs(
        raw,
        events,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False
    )

    print(f"Number of epochs: {len(epochs)}")

    # --------------------------------------------------------
    # Konwersja każdej epoki
    # --------------------------------------------------------

    all_epoch_data = epochs.get_data()

    subject_name = Path(edf_path).stem

    subject_output_dir = os.path.join(
        OUTPUT_DIR,
        subject_name
    )

    os.makedirs(subject_output_dir, exist_ok=True)

    for idx, epoch_data in enumerate(tqdm(all_epoch_data)):

        # epoch_data -> (channels, time)

        spectrograms = epoch_to_spectrogram(epoch_data)

        # zapis
        output_file = os.path.join(
            subject_output_dir,
            f"epoch_{idx:05d}.npy"
        )

        np.save(output_file, spectrograms)

    print("DONE")

# ============================================================
# MAIN
# ============================================================


def main():

    edf_files = []

    for file in os.listdir(RAW_DATA_DIR):

        if file.endswith(".edf"):
            edf_files.append(
                os.path.join(RAW_DATA_DIR, file)
            )

    print(f"Found EDF files: {len(edf_files)}")

    for edf_file in edf_files:
        process_edf(edf_file)

if __name__ == "__main__":
    main()
import os
from pathlib import Path

import mne
import numpy as np
import pywt
from tqdm import tqdm


# ============================================================
# KONFIGURACJA
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

RAW_DATA_DIR = str(PROJECT_DIR / "data" / "raw")
OUTPUT_DIR = str(PROJECT_DIR / "data" / "spectrograms")

# filtracja
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
NOTCH_FREQ = 50.0

# resampling
RESAMPLE_FREQ = 256

# wavelet
WAVELET = "morl"
SCALES = np.arange(1, 64)


# ============================================================
# FOLDERY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PREPROCESSING
# ============================================================


def preprocess_raw(raw):

    print("Filtering...")

    raw.filter(LOW_FREQ, HIGH_FREQ)

    raw.notch_filter(NOTCH_FREQ)

    raw.resample(RESAMPLE_FREQ)

    return raw

# ============================================================
# SPEKTROGRAM DLA JEDNEGO KANAŁU
# ============================================================


def create_wavelet_spectrogram(signal):

    coefficients, frequencies = pywt.cwt(
        signal,
        SCALES,
        WAVELET
    )

    power = np.abs(coefficients)

    return power.astype(np.float32)

# ============================================================
# NORMALIZACJA
# ============================================================


def normalize_spectrogram(spec):

    mean = np.mean(spec)
    std = np.std(spec)

    if std < 1e-6:
        std = 1e-6

    spec = (spec - mean) / std

    return spec.astype(np.float32)


# ============================================================
# KONWERSJA CAŁEGO EEG
# ============================================================


def eeg_to_spectrogram(raw_data):
    """
    Input:
        raw_data -> (channels, time)

    Output:
        spectrograms -> (channels, freq, time)
    """

    all_channel_specs = []

    for channel_signal in tqdm(raw_data):

        spec = create_wavelet_spectrogram(channel_signal)

        spec = normalize_spectrogram(spec)

        all_channel_specs.append(spec)

    spectrograms = np.stack(all_channel_specs)

    return spectrograms.astype(np.float32)

# ============================================================
# PROCESS EDF
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
    # Pobranie danych EEG
    # --------------------------------------------------------

    # shape -> (channels, time)
    raw_data = raw.get_data()

    print(f"Raw EEG shape: {raw_data.shape}")

    # --------------------------------------------------------
    # Tworzenie spektrogramów
    # --------------------------------------------------------

    spectrograms = eeg_to_spectrogram(raw_data)

    print(f"Spectrogram shape: {spectrograms.shape}")

    # --------------------------------------------------------
    # Zapis
    # --------------------------------------------------------

    filename = Path(edf_path).stem

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{filename}_spectrogram.npy"
    )

    np.save(output_path, spectrograms)

    print(f"Saved: {output_path}")

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
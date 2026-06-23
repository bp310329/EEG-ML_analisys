import os
import glob
from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================
# CONFIGURATION / KONFIGURACJA
# ==========================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Jesteśmy w Jupyter Notebook - bierzemy aktualny folder roboczy
    SCRIPT_DIR = Path.cwd()

# INTELIGENTNE SZUKANIE PROJEKTU:
# Jeśli folder 'data' jest bezpośrednio tutaj, to tu jest nasz ROOT.
# Jeśli nie, sprawdzamy poziom wyżej.
if (SCRIPT_DIR / "data").exists():
    PROJECT_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "data").exists():
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    # W ostateczności zakładamy domyślnie folder skryptu
    PROJECT_DIR = SCRIPT_DIR

# Używamy obiektów Path zamiast konwersji na str (glob i os.path świetnie z nimi współpracują)
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "Wyniki"
OUTPUT_DIR = PROJECT_DIR / "data" / "spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LINIE DIAGNOSTYCZNE (Dzięki temu zobaczysz, gdzie dokładnie patrzy Python) ---
print(f"Lokalizacja skryptu/notatnika: {SCRIPT_DIR}")
print(f"Wykryty główny folder projektu: {PROJECT_DIR}")
print(f"Ścieżka, w której szukam plików .edf: {RAW_DATA_DIR.resolve()}")
print(f"Czy ten folder fizycznie istnieje? {RAW_DATA_DIR.exists()}\n")
# ---------------------------------------------------------------------------------

# Częstotliwości do spektrogramu (zamiast skal, MNE operuje bezpośrednio na Hz)
FREQS = np.arange(1, 41, 1)  
N_CYCLES = FREQS / 2.0

# Słownik mapowania kategorii obrazków na unikalne ID dla modelu ML
CATEGORY_MAPPING = {'abstract': 101,
                    'airplane': 102, 
                    'apple': 103, 
                    'banana': 104, 
                    'bird': 105, 
                    'boat': 106,
                    'car': 107,
                    'dog': 108,
                    'person': 109,
                    'zebra': 110,
                    'train': 111}

# ==========================================================
# WYSZUKIWANIE PLIKÓW (używamy RAW_DATA_DIR jako Path)
# ==========================================================
edf_files = glob.glob(os.path.join(RAW_DATA_DIR, "*_raw.edf"))
print(f"Found EDF files: {len(edf_files)}")

# ==========================================================
# GŁÓWNA PĘTLA PRZETWARZANIA
# ==========================================================
for edf_file in edf_files:
    filename = os.path.basename(edf_file)
    subject_id = filename.replace('_raw.edf', '')
    
    print("\n" + "="*60)
    print(f"PROCESSING SUBJECT: {subject_id}")
    print("="*60)
    
    # Automatyczne dopasowanie pliku CSV z eventami
    csv_pattern = os.path.join(RAW_DATA_DIR, f"{subject_id}_EEGBasedVisualRecall_Events*.csv")
    csv_candidates = glob.glob(csv_pattern)
    
    if not csv_candidates:
        print(f"SKIPPING: Missing CSV events file for: {subject_id}")
        continue
    csv_file = csv_candidates[0]
    print(f"-> Matched CSV: {os.path.basename(csv_file)}")
    
    # ------------------------------------------------------
    # 1. WCZYTYWANIE I PREPROCESSING EEG (Twój kod)
    # ------------------------------------------------------
    raw = mne.io.read_raw_edf(edf_file, preload=True, infer_types=True, verbose=False)
    
    # Mapowanie i czyszczenie nazw kanałów
    mapping = {ch: ch.split('-')[0].split(':')[0].strip() for ch in raw.ch_names}
    raw.rename_channels(mapping)
    
    # USTWIENIE TYPÓW KANAŁÓW (POPRAWIONE)
    ch_types = {}
    for ch in raw.ch_names:
        # Dodajemy warunek .startswith('Imp'), aby odrzucić kanały impedancji
        if ch.startswith('Imp') or ch in ['X1', 'X2', 'X3', 'CM', 'Ax', 'Ay', 'Az']:
            ch_types[ch] = 'misc'  
        elif ch in ['Trigger', 'Event']:
            ch_types[ch] = 'stim'
        else:
            ch_types[ch] = 'eeg'
    raw.set_channel_types(ch_types)
    
    # Montaż systemowy 10-20
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    # Oznaczenie uszkodzonego kanału
    raw.info['bads'] = ['Pz']
    
    # Filtrowanie sygnału (1 - 40 Hz)
    raw.filter(1, 40, verbose=False)
    
    # Opcjonalne wyświetlanie wykresu PSD dla weryfikacji (możesz zakomentować w pętli)
    psd = raw.compute_psd(picks='eeg', exclude='bads', fmax=40, verbose=False)
    fig = psd.plot(show=False)
    fig.suptitle(f'PSD Cleaned – {subject_id}')
    plt.show()
    plt.close(fig) # Zamknięcie wykresu, aby nie przepełniać pamięci RAM
    
    # ------------------------------------------------------
    # 2. PARSOWANIE EVENTÓW I SINKRONIZACJA (MNE + CSV)
    # ------------------------------------------------------
    # Wyciągamy punkty czasowe '12' (IMAGE_ON) z adnotacji EDF
    event_id_map = {'12': 12}
    try:
        events, _ = mne.events_from_annotations(raw, event_id=event_id_map, verbose=False)
    except ValueError:
        print(f"SKIPPING: No '12' annotations found in EDF for {subject_id}")
        continue
        
    # Wczytanie etykiet z pliku CSV
    df = pd.read_csv(csv_file)
    df_images = df[df['event_code'] == 12].copy()
    
    # Sprawdzenie i korekta długości (w razie urwania sygnału)
    if len(events) != len(df_images):
        print(f"Warning: Event count mismatch (EDF: {len(events)}, CSV: {len(df_images)})")
        min_len = min(len(events), len(df_images))
        events = events[:min_len]
        df_images = df_images.iloc[:min_len]
        
    # Mapowanie tekstowych kategorii na ID numeryczne
    df_images['category_id'] = df_images['image_category'].map(CATEGORY_MAPPING)
    
    if df_images['category_id'].isnull().any():
        unknown = df_images[df_images['category_id'].isnull()]['image_category'].unique()
        print(f"ERROR: Unknown categories {unknown}. Update CATEGORY_MAPPING.")
        continue
        
    # Zastępujemy generyczny kod 12 konkretnymi ID klas (np. 101, 102)
    events[:, 2] = df_images['category_id'].values
    
    # ------------------------------------------------------
    # 3. TWORZENIE EPOK WOKÓŁ BODŹCA
    # ------------------------------------------------------
    # Automatycznie odrzucamy uszkodzone kanały ('bads') oraz kanały typu 'misc' i 'stim'
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=CATEGORY_MAPPING, 
        tmin=-0.2, 
        tmax=1.0, 
        baseline=(-0.2, 0.0), 
        picks='eeg',
        reject_by_annotation=True,
        preload=True,
        verbose=False
    )
    
    # ------------------------------------------------------
    # 4. GENEROWANIE SPEKTROGRAMÓW DLA MODELU ML (Falki Morleta)
    # ------------------------------------------------------
    print("-> Computing Wavelet Spectrograms (with decimation)...")
    
    # decim=4 zmniejszy rozdzielczość czasową spektrogramu 4-krotnie,
    # co drastycznie odciąży pamięć RAM, zachowując pełną informację do 40 Hz.
    tfr = mne.time_frequency.tfr_morlet(
        epochs, 
        freqs=FREQS, 
        n_cycles=N_CYCLES, 
        use_fft=True, 
        return_itc=False, 
        average=False, 
        decim=4,         # <-- KLUCZOWA POPRAWKA
        verbose=False
    )
    
    # Pobieramy macierz danych
    X = tfr.data
    y = epochs.events[:, 2]
    
    # Skalowanie decybelowe (dB)
    X = 10 * np.log10(X)
    
    # Konwersja z float64 na float32 - oszczędza kolejne 50% pamięci na dysku i w RAMie podczas trenowania
    X = X.astype(np.float32)
    
    # ------------------------------------------------------
    # 5. ZAPIS GOTOWYCH MACIERZY DLA MODELU ML
    # ------------------------------------------------------
    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"{subject_id}_y.npy"), y)
    
    print(f"SUCCESS: Saved data tensors for {subject_id}")
    print(f"   X shape (Spectrograms): {X.shape} -> [Epochs, Channels, Frequencies, Time]")
    print(f"   y shape (Labels): {y.shape}")
    
    # Czyszczenie pamięci po każdym badanym, żeby pętla nie puchła w trakcie działania
    del raw, epochs, tfr, X, y
print("\n All files processed successfully!")
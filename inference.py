import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. KONFIGURACJA I MAPOWANIE KLAS
# Słownik mapujący indeksy sieci (0-10) na czytelne nazwy obiektów
CLASS_NAMES = {
    0: "abstract", 1: "airplane", 2: "car", 3: "cat", 4: "dog",
    5: "face", 6: "fish", 7: "forest", 8: "motorcycle", 9: "scenery", 10: "train"
}

# Ścieżka do modeli oraz do danych testowych
MODELS_PATTERN = "./models/best_eegnet_fold_*.keras"
DATA_DIR = "./data/spectrograms"

# Wybierz indeks użytkownika do przetestowania (np. pierwszy plik z brzegu)
# Możesz tu wpisać konkretną nazwę pliku, np. os.path.join(DATA_DIR, "Subject_0_X.npy")
x_files = sorted(glob.glob(os.path.join(DATA_DIR, "mole_X.npy")))
if not x_files:
    raise FileNotFoundError(f"Nie znaleziono plików danych w katalogu {DATA_DIR}")

SELECTED_USER_X = x_files[0]  # Testujemy pierwszego użytkownika z listy
SELECTED_USER_Y = SELECTED_USER_X.replace('_X.npy', '_y.npy')

print(f"[INFO] Wybrany użytkownik do analizy: {os.path.basename(SELECTED_USER_X)}")

# 2. ŁADOWANIE KOMITETU MODELI (ENSEMBLE)
model_paths = sorted(glob.glob(MODELS_PATTERN))
if len(model_paths) == 0:
    raise FileNotFoundError("Nie znaleziono zapisanych modeli! Uruchom najpierw trening.")

print(f"[INFO] Znaleziono {len(model_paths)} modeli do stworzenia komitetu.")
models_committee = []
for path in model_paths:
    print(f" -> Ładowanie: {path}...")
    models_committee.append(load_model(path))

# 3. ŁADOWANIE I PREPROCESOWANIE DANYCH UŻYTKOWNIKA
X_user = np.load(SELECTED_USER_X)  # (Epoki, Kanały, Częstotliwości, Czas)
y_user = np.load(SELECTED_USER_Y)

# Dopasowanie układu osi pod TensorFlow (tak samo jak przy treningu)
X_user = np.transpose(X_user, (0, 2, 3, 1))  # (Epoki, Częstotliwość, Czas, Kanały)
y_user = y_user - 101                        # Skalowanie etykiet do zakresu 0-10

# Czyszczenie danych z potencjalnych błędów
X_user[np.isinf(X_user)] = 0.0
X_user[np.isnan(X_user)] = 0.0

# Normalizacja Z-score (lokalna dla tego użytkownika w celu unifikacji skali)
mean = X_user.mean()
std = X_user.std() if X_user.std() > 0 else 1.0
X_user = (X_user - mean) / std

# Opcjonalnie: jeśli chcesz przetestować tylko fragment danych (np. pierwsze 10 obrazów),
# odkomentuj poniższe dwie linijki:
# X_user = X_user[:10]
# y_user = y_user[:10]

num_samples = len(X_user)
print(f"[INFO] Dane przygotowane. Liczba obrazów do sklasyfikowania: {num_samples}\n")

# 4. ENSEMBLE LEARNING - INFERENCJA
print("="*60)
print(" ROZPOCZĘCIE KLASYFIKACJI (KOMITET MODELI) ")
print("="*60)

# Zbieramy predykcje (prawdopodobieństwa) z każdego modelu osobno
all_model_probabilities = []
for idx, model in enumerate(models_committee):
    # model.predict zwraca tablicę o kształcie (num_samples, 11)
    probs = model.predict(X_user, verbose=0)
    all_model_probabilities.append(probs)

# Zamieniamy listę na macierz numpy o kształcie (5_modeli, num_samples, 11_klas)
all_model_probabilities = np.array(all_model_probabilities)

# SOFT VOTING: Wyciągamy średnią arytmetyczną z prawdopodobieństw dla każdego obrazu
# Wynikowy kształt: (num_samples, 11_klas)
ensemble_probabilities = np.mean(all_model_probabilities, axis=0)

# Wybieramy klasę z najwyższym uśrednionym prawdopodobieństwem
final_predictions = np.argmax(ensemble_probabilities, axis=1)

# 5. ANALIZA WYNIKÓW DLA KAŻDEGO OBRAZU Z OSOBNA
correct_counts = 0

print(f"{'ID Obrazu':<10} | {'Prawdziwa klasa':<15} | {'Predykcja Komitetu':<18} | {'Status':<10} | {'Pewność sieci'}")
print("-" * 75)

for i in range(num_samples):
    true_idx = y_user[i]
    pred_idx = final_predictions[i]
    
    true_name = CLASS_NAMES.get(true_idx, f"Nieznana ({true_idx})")
    pred_name = CLASS_NAMES.get(pred_idx, f"Nieznana ({pred_idx})")
    
    # Wyciągamy procentową pewność wybranej klasy ze średniej komitetu
    confidence = ensemble_probabilities[i][pred_idx] * 100
    
    if true_idx == pred_idx:
        status = "OK"
        correct_counts += 1
    else:
        status = "BŁĄD"
        
    print(f"Obraz {i+1:<4} | {true_name:<15} | {pred_name:<18} | {status:<10} | {confidence:.1f}%")

# 6. PODSUMOWANIE KOŃCOWE
final_accuracy = (correct_counts / num_samples) * 100
print("=" * 75)
print(f"PODSUMOWANIE ANALIZY UŻYTKOWNIKA:")
print(f"Poprawnie sklasyfikowane: {correct_counts} / {num_samples}")
print(f"Ogólna celność komitetu modeli (Ensemble Accuracy): {final_accuracy:.2f}%")
print("=" * 75)
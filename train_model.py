import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import GroupKFold

# 1. PARAMETRY I ŚCIEŻKI
SPECTROGRAMS_DIR = "./data/spectrograms"
NUM_CLASSES = 11  # Masz 11 kategorii obrazków (abstract, airplane... train)

all_X = []
all_y = []
all_groups = []

# Ładowanie plików wygenerowanych w poprzednim kroku
x_files = glob.glob(os.path.join(SPECTROGRAMS_DIR, "*_X.npy"))

for subject_idx, x_path in enumerate(x_files):
    y_path = x_path.replace('_X.npy', '_y.npy')
    if not os.path.exists(y_path):
        continue
        
    X_sub = np.load(x_path)  # Kształt: (Epoki, Kanały, Częstotliwości, Czas)
    y_sub = np.load(y_path)
    
    # Zmieniamy układ osi pod TensorFlow
    X_sub = np.transpose(X_sub, (0, 2, 3, 1))
    y_sub = y_sub - 101
    
    # =================================================================
    # KROK RATUNKOWY: Zastępujemy wartości -inf oraz nan zwykłymi zerami
    # =================================================================
    X_sub[np.isinf(X_sub)] = 0.0
    X_sub[np.isnan(X_sub)] = 0.0
    # =================================================================
    
    all_X.append(X_sub)
    all_y.append(y_sub)
    all_groups.append(np.full(len(y_sub), subject_idx))

# Łączenie danych wszystkich użytkowników
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
groups = np.concatenate(all_groups, axis=0)

print(f"Dane załadowane! Kształt X dla TensorFlow: {X.shape}") # [Epoki, Częstotliwość, Czas, Kanały]
print(f"Kształt y: {y.shape}")

# 2. DEFINICJA ARCHITEKTURY SIECI CNN
def build_cnn_model(input_shape, num_classes):
    """Mocno zregularyzowana sieć CNN dedykowana do trudnych danych EEG."""
    reg = tf.keras.regularizers.l2(0.001) # Kara L2 zapobiegająca przeuczeniu
    
    model = models.Sequential([
        # Warstwa 1
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2), # <-- Wycinamy całe mapy cech, a nie pojedyncze piksele
        layers.MaxPooling2D((2, 2)),
        
        # Warstwa 2
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2, 2)),
        
        # Warstwa 3
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(), 
        
        # Klasyfikator z wyższym Dropoutem
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5), # Podnosimy do 50%
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Zaczynamy wyżej, bo scheduler sam go zmniejszy
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. PĘTLA TRENINGOWA (SUBJECT-INDEPENDENT)
gkf = GroupKFold(n_splits=5) # Dzielimy zbiór na 5 części (całymi ludźmi)
input_shape = (X.shape[1], X.shape[2], X.shape[3]) # (Częstotliwości, Czas, Kanały)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n" + "="*50)
    print(f" TRENOWANIE FOLD {fold + 1} / 5")
    print("="*50)
    
    # Podział na zbiór treningowy i testowy (nowe osoby!)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Normalizacja danych (Z-score) bazowana na danych treningowych
    mean = X_train.mean()
    std = X_train.std()

    X_train -= mean
    X_train /= std
    
    X_test -= mean
    X_test /= std
    
    # Tworzymy czysty model dla tego foldu
    model = build_cnn_model(input_shape, NUM_CLASSES)

    # Definicja Early Stoppingu
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=6,                  # Jeśli przez 8 epok val_loss nie spadnie, kończymy
        restore_best_weights=True    # Zachowaj najlepszą wersję modelu, a nie tę zepsutą z końca
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,       # Zmniejsz krok uczenia o połowę (np. z 0.001 na 0.0005)
        patience=3,       # Jeśli przez 3 epoki val_loss nie spada, reaguj
        verbose=1,
        min_lr=1e-5
    )
    
    # URUCHOMIENIE TRENINGU
    # epochs=15 oznacza, że model przejrzy dane treningowe 15 razy. 
    # batch_size=32 oznacza, że aktualizuje wagi sieci po każdych 32 spektrogramach.
    history = model.fit(
        X_train, y_train, 
        epochs=50,                  # Zwiększamy sufit do 50 epok
        batch_size=32, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping, lr_scheduler],  # <-- Wstrzykujemy naszego strażnika
        verbose=1
    )
    
    # Ewaluacja na osobach odłożonych do testu
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n-> Wynik FOLD {fold + 1}: Celność na NIEZNANYCH użytkownikach = {test_acc*100:.2f}%")
    
    # Zatrzymujemy pętlę po pierwszym foldzie na próbę, żebyś zobaczył czy działa
    print("\n[INFO] Próba udana. Jeśli chcesz przetestować wszystkie foldy, usuń instrukcję 'break'.")
    break
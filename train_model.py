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
    
    # Zmieniamy układ osi pod TensorFlow -> (Epoki, Częstotliwość, Czas, Kanały)
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

print(f"Dane załadowane! Kształt X dla TensorFlow: {X.shape}") 
print(f"Kształt y: {y.shape}")


# 2. DEFINICJA ARCHITEKTURY EEGNet (Dostosowanej do Spektrogramów)
def build_eegnet_model(input_shape, num_classes):
    F1 = 8  
    D = 2   
    F2 = F1 * D  
    num_channels = input_shape[2] 
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Permute((3, 2, 1)),
        
        # BLOCK 1
        layers.Conv2D(filters=F1, kernel_size=(1, 15), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D(kernel_size=(num_channels, 1), padding='valid', 
                               depth_multiplier=D, use_bias=False,
                               depthwise_constraint=tf.keras.constraints.MaxNorm(1.0)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D(pool_size=(1, 4)),
        layers.Dropout(0.25),
        
        # BLOCK 2
        layers.SeparableConv2D(filters=F2, kernel_size=(1, 16), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D(pool_size=(1, 4)),
        layers.Dropout(0.25),
        
        # KLASYFIKATOR
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax', 
                     kernel_constraint=tf.keras.constraints.MaxNorm(0.25))
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# 3. PĘTLA TRENINGOWA (SUBJECT-INDEPENDENT)
gkf = GroupKFold(n_splits=5) 
input_shape = (X.shape[1], X.shape[2], X.shape[3]) 

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n" + "="*50)
    print(f" TRENOWANIE FOLD {fold + 1} / 5 (Model: EEGNet)")
    print("="*50)
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Normalizacja danych (Z-score)
    mean = X_train.mean()
    std = X_train.std()
    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std
    
    # Tworzymy czysty model
    model = build_eegnet_model(input_shape, NUM_CLASSES)

    # --- DEFINICJA CALLBACKÓW (STRAŻNIKÓW) ---
    
    # 1. Early Stopping - przerwie gdy model zacznie "puchnąć" z overfittingu
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20,                  
        restore_best_weights=True    
    )

    # 2. ReduceLROnPlateau - zmniejszy krok uczenia w trudnych momentach
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,       
        patience=3,       
        verbose=1,
        min_lr=1e-5
    )
    
    # 3. NOWOŚĆ: ModelCheckpoint - automatyczny zapis najlepszych wag na dysk
    model_filename = f"models/best_eegnet_fold_{fold + 1}.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_filename,
        monitor='val_loss',         # Obserwujemy stratę na nowym użytkowniku
        save_best_only=True,        # Zapisuj TYLKO wtedy, gdy pobiliśmy rekord (najniższy val_loss)
        mode='min',                 # Interesuje nas minimalizacja straty
        verbose=1                   # Wyświetli komunikat w konsoli, kiedy plik zostanie nadpisany
    )
    
    # URUCHOMIENIE TRENINGU (Przekazujemy komplet 3 callbacków)
    history = model.fit(
        X_train, y_train, 
        epochs=50,                  
        batch_size=32, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping, lr_scheduler, checkpoint],  # <-- Dodany checkpoint
        verbose=1
    )
    
    # Ewaluacja na osobach odłożonych do testu (wykorzystuje przywrócone najlepsze wagi)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n-> Wynik FOLD {fold + 1}: Celność na NIEZNANYCH użytkownikach = {test_acc*100:.2f}%")
    print(f"[INFO] Najlepsza wersja tego modelu jest bezpieczna w pliku: {model_filename}")
    
    print("\n[INFO] Próba udana. Jeśli chcesz przetestować wszystkie foldy, usuń instrukcję 'break'.")
    #break
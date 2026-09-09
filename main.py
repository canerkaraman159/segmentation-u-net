import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# veri hazirlama ve yukleme
def load_dataset(root, img_size=(128, 128)):
    images, masks = [], []
    for tile in sorted(os.listdir(root)):
        img_dir = os.path.join(root, tile, "images")
        mask_dir = os.path.join(root, tile, "masks")
        if not os.path.isdir(img_dir): continue
        for f in os.listdir(img_dir):
            if not f.lower().endswith(".jpg"): continue
            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
            if not os.path.exists(mask_path): continue

            # goruntuyu oku ve rgbye cevir
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0

            # maskeyi gri tonlamada oku yeniden boyutlandir ve normalize et
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = np.expand_dims(mask, axis=-1) / 255.0

            images.append(img)
            masks.append(mask)

    return np.array(images, dtype="float32"), np.array(masks, dtype="float32")

# veriyi yukle ve ornek sayisini yazdir
X, y = load_dataset("aerial_dataset", img_size=(128, 128))
print(f"Toplam ornek: {len(X)}")

# egitim ve dogrulama setlerine ayir
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(f"Toplam train ornek: {len(X_train)}")
print(f"Toplam test ornek: {len(X_val)}")


# unet mimarisi tanimlama
def unet_model(input_size=(128, 128, 3)):
    inputs = keras.Input(input_size)

    # encoder: feature extraction ve downsampling
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)  # input -> inputs düzeltildi
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck: en derin seviye
    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(p4)
    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(c5)
    
    # decoder: up sampling ve skip connection
    u6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(c9)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)  # Con2D -> Conv2D düzeltildi

    return keras.Model(inputs=inputs, outputs=outputs)  # Mode -> Model düzeltildi


# egitim asamasi
model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("model_best.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    callbacks=callbacks
)

#sonuç değerlenidmre

import matplotlib.pyplot as plt

# sonuclarin degerlendirilmesi
plt.plot(history.history["loss"], label = "train_loss") # egitim kaybi
plt.plot(history.history["val_loss"], label = "val_loss") # dogrulama kaybi
plt.legend()
plt.show()

def show_prediction(idx = 0):
    img = X_val[idx]
    mask_true = y_val[idx].squeeze() # gercek maske
    pred_raw = model.predict(img[None, ...])[0].squeeze() # modelden tahmini al ve kanali sikistir
    mask_pred = (pred_raw > 0.5).astype("float32") # 0.5 esik degeri ile 2 li maske olustur

    # sonuclari gorsellestir
    plt.figure(figsize = (10, 4))
    
    # 1. Orijinal Girdi
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input")
    plt.axis("off")

    # 2. Gerçek Maske (Ground Truth)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_true, cmap="gray")
    plt.title("True Mask")
    plt.axis("off")

    # 3. Modelin Tahmini (Prediction)
    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Örnek bir doğrulama verisi üzerinde çalıştırma
show_prediction(1)
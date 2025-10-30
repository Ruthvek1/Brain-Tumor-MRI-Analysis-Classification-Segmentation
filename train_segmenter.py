import os
import numpy as np
import cv2  # OpenCV
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar

# --- Configuration ---
IMG_WIDTH = 256  # U-Net works well with powers of 2
IMG_HEIGHT = 256
IMG_CHANNELS = 3

# --- TODO: UPDATE THESE PATHS ---
DATA_ROOT_DIR = 'Brain-Tumor-Dataset'
TRAIN_IMG_DIR = os.path.join(DATA_ROOT_DIR, 'Training')
TRAIN_MASK_DIR = os.path.join(DATA_ROOT_DIR, 'Tumor-Mask')
# ------------------------------

# Folders that have masks
TUMOR_CLASSES_WITH_MASKS = ['glioma', 'meningioma', 'pituitary_tumor']


def find_mask_file(mask_folder, img_name):
    """
    Smartly finds the matching mask file.
    Handles identical names (G1.jpg -> G1.jpg)
    Handles prefix names (G1.jpg -> tumor mask G1.jpg)
    Handles different extensions (.png / .jpg)
    """
    base_name, img_ext = os.path.splitext(img_name)

    # 1. Check for identical name
    mask_path = os.path.join(mask_folder, img_name)
    if os.path.exists(mask_path):
        return mask_path

    # 2. Check for prefix name (from user's upload)
    mask_path = os.path.join(mask_folder, f"tumor mask {img_name}")
    if os.path.exists(mask_path):
        return mask_path

    # 3. Check for identical name but different extension
    for ext in ['.png', '.jpg', '.jpeg', '.tif']:
        mask_path = os.path.join(mask_folder, f"{base_name}{ext}")
        if os.path.exists(mask_path):
            return mask_path

    # 4. Check for prefix name but different extension
    for ext in ['.png', '.jpg', '.jpeg', '.tif']:
        mask_path = os.path.join(mask_folder, f"tumor mask {base_name}{ext}")
        if os.path.exists(mask_path):
            return mask_path

    # If no mask is found
    return None


def load_data(img_height, img_width):
    """
    Loads and preprocesses image and mask pairs.
    """
    images = []
    masks = []
    print(f"--- [Segmenter] Loading data from {DATA_ROOT_DIR} ---")

    for tumor_type in TUMOR_CLASSES_WITH_MASKS:
        img_folder = os.path.join(TRAIN_IMG_DIR, tumor_type)
        mask_folder = os.path.join(TRAIN_MASK_DIR, tumor_type)

        if not os.path.exists(img_folder):
            print(f"Warning: Image folder not found: {img_folder}")
            continue
        if not os.path.exists(mask_folder):
            print(f"Warning: Mask folder not found: {mask_folder}")
            continue

        img_filenames = os.listdir(img_folder)

        print(f"\n--- [Segmenter] Loading {tumor_type} images... ---")
        found_pairs = 0
        for img_name in tqdm(img_filenames):
            try:
                img_path = os.path.join(img_folder, img_name)
                mask_path = find_mask_file(mask_folder, img_name)

                if not os.path.exists(img_path) or mask_path is None:
                    continue  # Skip if image is broken or no mask was found

                # Load image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None: continue  # Skip broken images
                img = cv2.resize(img, (img_width, img_height))
                img = img / 255.0  # Normalize

                # Load mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None: continue  # Skip broken masks
                mask = cv2.resize(mask, (img_width, img_height))
                mask = (mask > 127).astype(np.float32)  # Binarize (0 or 1)
                mask = np.expand_dims(mask, axis=-1)  # Add channel dim

                images.append(img)
                masks.append(mask)
                found_pairs += 1
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
        print(f"Found {found_pairs} matching image/mask pairs for {tumor_type}.")

    print(f"\n--- [Segmenter] Loaded {len(images)} total image/mask pairs. ---")
    return np.array(images), np.array(masks)


def build_unet(input_shape):
    """
    Builds a simple U-Net model for segmentation.
    """
    print("--- [Segmenter] Building U-Net Model... ---")
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)

    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])  # Skip connection
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])  # Skip connection
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])  # Skip connection
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs], name="U-Net_Segmenter")

    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=BinaryCrossentropy(),
        metrics=[dice_coefficient]
    )
    model.summary()
    return model


def main():
    print("--- Starting Segmenter Training ---")

    # Check if paths are set
    if 'path/to/your' in DATA_ROOT_DIR:
        print("=" * 50)
        print("ERROR: Please update DATA_ROOT_DIR in this script.")
        print("=" * 50)
        return

    images, masks = load_data(IMG_HEIGHT, IMG_WIDTH)

    if len(images) == 0:
        print("No data loaded. Please check your file paths and filenames.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} images, validating on {len(X_val)} images.")

    model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    print("\n--- [Segmenter] Starting Model Training... ---")
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=30,  # Increase for better results (50+)
        validation_data=(X_val, y_val)
    )

    print("\n--- [Segmenter] Training complete. ---")

    # Save model
    model_filename = 'segmenter_model.h5'
    model.save(model_filename)
    print(f"Model saved as '{model_filename}'")

    print("\n--- [Segmenter] Evaluating model on validation set... ---")
    loss, dice = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Dice Coefficient: {dice:.4f}")
    print("--- [Segmenter] Script Finished. ---")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)

    main()


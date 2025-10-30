import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_ROOT_DIR = 'Brain-Tumor-Dataset'
TRAIN_DIR = os.path.join(DATA_ROOT_DIR, 'Training')


INCEPTION_IMG_WIDTH = 299
INCEPTION_IMG_HEIGHT = 299
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 20


def load_data_generators():
    """
    Sets up the ImageDataGenerators for classification.
    """
    print("--- [Classifier] Setting up Data Generators... ---")

    # Training Data Generator with Augmentation
    # We add 'validation_split=0.2' to use 20% of data for validation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        validation_split=0.2 
    )


    # --- Generators for InceptionV3 ---
    # We specify 'subset='training'' to get the 80%
    inception_train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(INCEPTION_IMG_WIDTH, INCEPTION_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'  
    )

    # We create a new validation generator from the 20% split
    inception_validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, 
        target_size=(INCEPTION_IMG_WIDTH, INCEPTION_IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',  
        shuffle=False 
    )

    print("Class Indices:", inception_train_generator.class_indices)
    # We now return the validation generator
    return inception_train_generator, inception_validation_generator


def build_inceptionv3_model():
    """Builds the InceptionV3 model from your notebook."""
    print("--- [Classifier] Building InceptionV3 Model... ---")
    inception_base = InceptionV3(
        input_shape=(INCEPTION_IMG_WIDTH, INCEPTION_IMG_HEIGHT, 3),
        include_top=False,
        weights='imagenet'
    )
    inception_base.trainable = False

    model = Sequential([
        inception_base,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ], name="InceptionV3_Classifier")

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def main():
    """Main function to run the training and evaluation."""

    print("--- Starting Classifier Training ---")

    # Check if paths are set
    if 'path/to/your' in DATA_ROOT_DIR:
        print("=" * 50)
        print("ERROR: Please update DATA_ROOT_DIR in this script.")
        print("=" * 50)
        return

    train_generator, validation_generator = load_data_generators()

    model = build_inceptionv3_model()

    print("\n--- [Classifier] Starting Model Training... ---")
    print(f"Training for {EPOCHS} epochs.")

    history = model.fit(
        train_generator,
        
        
        steps_per_epoch=int(np.ceil(train_generator.samples / BATCH_SIZE)),
        validation_data=validation_generator,
        validation_steps=int(np.ceil(validation_generator.samples / BATCH_SIZE)),
        epochs=EPOCHS
    )

    print("\n--- [Classifier] Training complete. ---")

    model_filename = 'classifier_model.h5'
    model.save(model_filename)
    print(f"Model saved as '{model_filename}'")

    print("\n--- [Classifier] Evaluating model on validation set... ---")
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\n--- [Classifier] Generating classification report... ---")
    # We predict on the validation_generator
    # NOTE: We pass the generator itself, Keras handles the steps
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())

    
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)
    print("--- [Classifier] Script Finished. ---")


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


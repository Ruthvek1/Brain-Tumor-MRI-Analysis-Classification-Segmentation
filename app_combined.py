import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2  # OpenCV
import io
import os

# --- Configuration ---
CLASSIFIER_MODEL_PATH = 'classifier_model.h5'
SEGMENTER_MODEL_PATH = 'segmenter_model.h5'

# --- Classifier Config ---
CLASS_IMG_WIDTH = 299
CLASS_IMG_HEIGHT = 299
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# --- Segmenter Config ---
SEG_IMG_WIDTH = 256
SEG_IMG_HEIGHT = 256


# --- Custom Metric for Segmenter ---
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


# --- Model Loading Functions ---

@st.cache_resource
def load_classifier_model():
    """Loads the InceptionV3 classifier model."""
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        return None
    model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    return model


@st.cache_resource
def load_segmenter_model():
    """Loads the U-Net segmenter model."""
    if not os.path.exists(SEGMENTER_MODEL_PATH):
        return None
    model = tf.keras.models.load_model(
        SEGMENTER_MODEL_PATH,
        custom_objects={'dice_coefficient': dice_coefficient}
    )
    return model


# --- Preprocessing & Prediction Functions ---

def preprocess_for_classifier(image_pil):
    """Prepares image for InceptionV3."""
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    img = image_pil.resize((CLASS_IMG_WIDTH, CLASS_IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def preprocess_for_segmenter(image_pil):
    """Prepares image for U-Net."""
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    img = image_pil.resize((SEG_IMG_WIDTH, SEG_IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def get_segmentation_results(model, image_pil):
    """Gets the mask, highlighted image, and size."""
    # 1. Get prediction (at 256x256)
    processed_image = preprocess_for_segmenter(image_pil)
    pred_mask = model.predict(processed_image)[0]
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255

    # 2. Overlay mask on original image
    original_cv = np.array(image_pil.convert('RGB'))
    # Resize mask to the original image's size for accurate display
    mask_resized = cv2.resize(pred_mask_binary, (original_cv.shape[1], original_cv.shape[0]))

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    highlighted_image = original_cv.copy()
    cv2.drawContours(highlighted_image, contours, -1, (255, 0, 0), 2)  # Draw in red

    # 3. Calculate size
    tumor_pixels = np.sum(mask_resized == 255)

    return highlighted_image, tumor_pixels


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Brain Tumor Analyzer")
st.title("🧠 Brain Tumor MRI Analyzer (Classifier + Segmenter)")
st.markdown("Upload an MRI scan to get **both** the tumor type and its location/size.")


pixel_spacing_mm = 0.10

# Load models
classifier_model = load_classifier_model()
segmenter_model = load_segmenter_model()

# Check if models are loaded
if classifier_model is None or segmenter_model is None:
    st.error("One or more models are missing!")
    st.info(f"Please run `train_classifier.py` to create `{CLASSIFIER_MODEL_PATH}`.")
    st.info(f"Please run `train_segmenter.py` to create `{SEGMENTER_MODEL_PATH}`.")
else:
    st.success("All models loaded successfully!")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(io.BytesIO(uploaded_file.read()))

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image, caption="Original MRI Scan", use_column_width=True)

        with col2:
            with st.spinner("Analyzing image... This may take a moment."):

                # --- Task 1: Classification ---
                class_image = preprocess_for_classifier(original_image)
                class_pred = classifier_model.predict(class_image)
                class_name = CLASS_NAMES[np.argmax(class_pred)]
                confidence = np.max(class_pred) * 100

                # --- Task 2: Segmentation ---
                highlighted_image, tumor_pixels = get_segmentation_results(segmenter_model, original_image)

                area_per_pixel_mm2 = pixel_spacing_mm * pixel_spacing_mm

                tumor_area_mm2 = tumor_pixels * area_per_pixel_mm2

                tumor_area_cm2 = tumor_area_mm2 / 100

                # --- Display Results ---
                st.image(highlighted_image, caption="Scan with Highlighted Tumor", use_column_width=True)

                st.subheader("Final Diagnosis:")

                # Show classification result
                st.metric(label="Predicted Tumor Type", value=class_name)
                st.write(f"Confidence: {confidence:.2f}%")

                # Show segmentation result
                if class_name != "No Tumor" and tumor_pixels > 10:

                    st.metric(label="Estimated Tumor Area", value=f"{tumor_area_cm2:.2f} cm²")

                elif class_name != "No Tumor":
                    st.warning("Tumor classified, but segmentation model could not find a distinct area.")
                else:
                    st.success("No tumor was classified.")




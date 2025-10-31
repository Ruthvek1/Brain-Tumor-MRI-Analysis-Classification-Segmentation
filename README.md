# Brain Tumor MRI Analysis: Classification & Segmentation

The project is a one-stop solution to analysing brain tumor MRI scans. It employs a two model deep learning pipeline to carry out two important functions:

1.  **Classification**: Determines the type of tumor of one of four categories: Glioma, Meningioma, Pituitary Tumor.
2.  **Segmentation**: Creates a pixel perfect mask to indicate the precise location and extent of the tumor, and computes its area in cm 2.

All the pipeline is enclosed by an easy-to-use Streamlit web application.

<img width="600" height="400" alt="mri" src="https://github.com/user-attachments/assets/6f4d5cab-8690-42b5-973e-76fa4fd2ee8d" />
<br>
<br>
<img width="600" height="400" alt="Streamlit MRI" src="https://github.com/user-attachments/assets/ad50693d-1516-481d-aef9-a8aa4847f12e" />

Screenshot of the Streamlit App showing a brain MRI with a red outline around the tumor and the classification 'Glioma Tumor'




## The Problem

Brain tumors are a life-threatening condition, and their early and accurate diagnosis is critical for effective treatment planning. Radiologists manually analyze hundreds of MRI scans, a process that can be time-consuming and subjective. An automated system that can accurately classify the tumor type and precisely outline its location and size can act as a powerful "second opinion" for medical professionals, leading to faster and more reliable diagnoses.

## Dataset

This project utilizes the Brain Tumor MRI Dataset from Figshare. This dataset contains 3,459 T1-weighted contrast-enhanced MRI images. The dataset is divided into four classes: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

A crucial part of this dataset is the inclusion of a Tumor-Mask folder, which provides ground-truth segmentation masks for 3,064 of the images (for the three tumor classes). This allows for supervised training of both classification and segmentation models.

## Data Preprocessing

The data was preprocessed differently for each model to meet its specific input requirements:

* **For Classification (InceptionV3):**
    * Images were resized to $299 \times 299$ pixels.
    * Data was split $80/20$ into training and validation sets using the `validation_split` feature of Keras.
    * Training images were augmented (rotation, zoom, shear, flip) to create a more robust model.
    * Pixel values were rescaled from $[0, 255]$ to $[0, 1]$.

* **For Segmentation (U-Net):**
    * Images and their corresponding masks were resized to $256 \times 256$ pixels.
    * Images were rescaled to $[0, 1]$.
    * Masks were loaded as grayscale and binarized (pixels > 127 became 1, all others 0) to create a clear binary target.

## Methods

This project implements a dual-model pipeline, where each model is a specialized Convolutional Neural Network (CNN) optimized for its specific task.

![Pipeline Diagram](assets/pipeline.png)

### 1. Classification Model: InceptionV3 (Transfer Learning)

To identify the tumor type, this project employs transfer learning using the InceptionV3 architecture.

* **Why InceptionV3?** Instead of training a CNN from scratch, we use a model pre-trained on the massive ImageNet dataset. This model already possesses a powerful understanding of shapes, textures, and features. We "freeze" these layers and only train a new "head" (a few Dense layers) to adapt this knowledge to our specific task of classifying brain tumors. This is significantly faster and more accurate than training a smaller model from scratch.

### 2. Segmentation Model: U-Net

To find the tumor's location and size, this project implements a U-Net model.

* **Why U-Net?** The U-Net is the industry standard for biomedical image segmentation. Its unique encoder-decoder architecture with "skip connections" allows it to capture fine-grained spatial information (from the encoder) and combine it with high-level contextual information (from the decoder). This makes it exceptionally good at outlining complex shapes like tumors with high precision.

![U-Net Architecture](assets/unet.png)

* **How it works:** The model takes a $256 \times 256$ MRI scan as input and outputs a $256 \times 256$ binary "mask," where each pixel is classified as either "tumor" (1) or "not tumor" (0). This mask is then used to draw the contour on the original image and calculate the area.

## Steps to Run the Code

This project is built in Python 3.9.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Ruthvek1/brain-tumor-app.git](https://github.com/Ruthvek1/brain-tumor-app.git)
    cd brain-tumor-app
    ```

2.  **Create a Virtual Environment & Install Dependencies**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set Data Paths**
    Before training, you must edit the `DATA_ROOT_DIR` variable at the top of both training scripts to point to your local dataset.
    * `train_classifier.py`
    * `train_segmenter.py`

4.  **Run Training Scripts**
    You must train the models, which will save the `.h5` files needed by the app.
    ```bash
    # This trains the classifier and saves classifier_model.h5
    python train_classifier.py
    
    # This trains the segmenter and saves segmenter_model.h5
    python train_segmenter.py
    ```
    Note: If you are on a Mac, you may need to downgrade NumPy to version 1.26.4 to run training (`pip install numpy==1.26.4`).

5.  **Run the Streamlit App**
    * **On macOS:** You must use this command to avoid a mutex lock crash:
        ```bash
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES streamlit run app_combined.py
        ```
    * **On Windows/Linux:**
        ```bash
        streamlit run app_combined.py
        ```

## Experiments & Results

### Classification Performance

Early experiments were conducted to compare different architectures for the classification task. A simple Multi-Layer Perceptron (MLP) and a standard CNN (AlexNet) were trained and compared against the InceptionV3 transfer learning model. The results clearly show that the InceptionV3 model is the superior choice, achieving 82.49% accuracy.

| Model | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- |
| MLP | ~62% | ~0.37 |
| AlexNet | ~61% | ~0.38 |
| **InceptionV3** | **82.49%** | **0.83** |

(Note: F1-Score from the original notebook was low due to an evaluation error; the 0.83 score is from the final, corrected script's classification report.)

### Segmentation & Combined Results

The U-Net model successfully learns to produce precise binary masks for the tumor regions. The final Streamlit application successfully combines both models into a single diagnostic tool. The app provides the classification (e.g., "Glioma Tumor") and displays the segmented mask overlaid on the original image. It then uses the pixel count of the mask to provide a quantitative size calculation.

| Original Image | Ground Truth Mask | Final App Prediction |
| :---: | :---: | :---: |
| ![Original MRI](assets/test_image_1.jpg) | ![Ground Truth Mask](assets/test_mask_1.jpg) | ![App Prediction](assets/app_prediction_1.png) |
| ![Original MRI](assets/test_image_2.jpg) | ![Ground Truth Mask](assets/test_mask_2.jpg) | ![App Prediction](assets/app_prediction_2.png) |

## Conclusion

This project successfully demonstrates the power of a dual-CNN pipeline for comprehensive brain tumor analysis.

* **Key Finding 1:** For classification, transfer learning with InceptionV3 is vastly superior to training smaller models from scratch, achieving over 20% higher accuracy.
* **Key Finding 2:** A U-Net model can be effectively trained on the provided mask data to produce accurate segmentation, which is the key to moving from a qualitative ("there is a tumor") to a quantitative ("the tumor is 2.14 cm²") analysis.

The final Streamlit application serves as a powerful and easy-to-use proof-of-concept for a tool that could one day assist radiologists by automating and accelerating the diagnostic process.

## References

1.  Cheng, J. (2017). brain tumor dataset. Figshare. https://doi.org/10.6084/m9.figshare.1512427.v5
2.  Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
3.  Szegedy, C., Vanhoucke, V., Ioffe, S., et al. (2016). Rethinking the Inception Architecture for Computer Vision.
4.  Chahal, P. K., Pandey, S., & Goel, S. (2020). A Survey On Brain Tumor Detection Techniques For MR images.
5.  Pereira, S., Pinto, A., Alves, V., & Silva, C. A. (2016). Brain Tumor Segmentation Using Convolutional Neural Networks In MRI Images.
6.  Cheng, J., Huang, W., Cao, S., et al. (2015). Enhanced Performance of Brain Tumor Classification Via Tumor Region Augmentation & Partition.
7.  Cheng, J., Yang, W., Huang, M., et al. (2016). Retrieval of Brain Tumors By Adaptive Spatial Pooling & Fisher Vector Representation.
8.  Deepa, A.R., & Sam Emmanuel, W.R. (2019). A Comprehensive Review And Analysis On MRI Based Brain Tumor Segmentation.

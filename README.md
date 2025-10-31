# Brain Tumor MRI Analysis: Classification & Segmentation

The project is a one-stop solution to analysing brain tumor MRI scans. It employs a two model deep learning pipeline to carry out two important functions:

1.  **Classification**: Determines the type of tumor of one of four categories: Glioma, Meningioma, Pituitary Tumor.
2.  **Segmentation**: Creates a pixel perfect mask to indicate the precise location and extent of the tumor, and computes its area in cm 2.

All the pipeline is enclosed by an easy-to-use Streamlit web application.

<img width="600" height="400" alt="mri" src="https://github.com/user-attachments/assets/6f4d5cab-8690-42b5-973e-76fa4fd2ee8d" />
<br>
<br>
<br>
<img width="600" height="400" alt="Streamlit MRI" src="https://github.com/user-attachments/assets/ad50693d-1516-481d-aef9-a8aa4847f12e" />

Screenshot of the Streamlit App showing a brain MRI with a red outline around the tumor and the classification 'Glioma Tumor'




## The Problem

The dataset used in this project is the Brain Tumor MRI Dataset of Figshare. This data comprised 3459 T1-weighted contrast enhanced MRI images. The data set will be classified into four categories namely Glioma, Meningioma, Pituitary Tumor.

An important aspect of such a dataset is that it includes a Tumor-Mask folder, a ground-truth segmentation mask of 3064 of the images (one of three types of tumor). This enables a supervised training of classification as well as segmentation models.

## Dataset

This project utilizes the Brain Tumor MRI Dataset from Figshare. This dataset contains 3,459 T1-weighted contrast-enhanced MRI images. The dataset is divided into four classes: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

A crucial part of this dataset is the inclusion of a Tumor-Mask folder, which provides ground-truth segmentation masks for 3,064 of the images (for the three tumor classes). This allows for supervised training of both classification and segmentation models.

## Data Preprocessing

Each model had a different preprocessing of the data to fit the input of the model:

* **For Classification (InceptionV3):**
    * Images were made $299 \times 299$ pixels.
    * The `validation_split` feature of Keras was used to split the data in 80/20 into training and validation sets.
    * Images used as training were augmented (rotation, zoom, shear, flip) to form a more robust model.
    * The rescaling of pixel values was done from $[0, 255]$ to $[0, 1]$.

* **For Segmentation (U-Net):**
    * The images and the masked images were scaled to $256 \times 256$ pixels.
    * Images were rescaled to $[0, 1]$.
    * The masks were loaded and converted to grayscale and binarized (pixels > 127 became 1, all others 0) to have a distinct binary target.

## Methods

This project uses a dual model pipeline each of which is a specialized Convolutional Neural Network (CNN) that is optimized to perform its job.
<br>
<br>
<img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/e207cf7c-4930-4b82-a0d8-3be7c78278f4" />


### 1. Classification Model: InceptionV3 (Transfer Learning)

In this project, transfer learning with InceptionV3 architecture is used in order to diagnose the type of the tumor.

* **Why InceptionV3?** We do not need to train a CNN ourselves, using one that has already been trained on the giant ImageNet dataset. This model already has a strong conceptualisation of shapes, textures and features. These layers are frozen and we only need to train a new "head" (some Dense layers) to fit this knowledge to our task of classifying brain tumors. It is much faster and more precise than training a smaller model blanketed.


### 2. Segmentation Model: U-Net

In order to locate the location and the size of the tumor, this project uses a U-Net model.

* **Why U-Net?** The U-Net is biomedical image segmentation available in the industry. Its encoder-decoder structure with skip connections makes it possible to accumulate fine-grained spatial data (via the encoder) to be processed with high-level contextual data (via the decoder). This gives it a very high level of precision in defining complex shapes such as tumors.

<img width="500" height="500" alt="Gemini_Generated_Image_lbssfalbssfalbss" src="https://github.com/user-attachments/assets/0f3d71ad-3841-434f-b3d8-788bbc76c82a" /><br>


**How it works:** The model is fed with a $256 \times 256$ MRI scan and outputs a $256 \times 256$ binary "mask" with pixels that are either labeled as tumor 1 or not tumor 0.The mask is then applied to the original image in order to obtain the contour and compute the area.

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

The classification task was experimented using various different architectures at the beginning. An inceptionV3 transfer learning model was compared to a simple Multi-Layer Perceptron (MLP), as well as to a typical CNN (AlexNet).
The findings indicate clearly that the InceptionV3 model is the better option, as it had a score of 82.49%.

| Model | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- |
| MLP | ~62% | ~0.37 |
| AlexNet | ~61% | ~0.38 |
| **InceptionV3** | **82.49%** | **0.83** |


### Segmentation & Combined Results


The U-Net model is effective in learning to give accurate binary masks of the tumor regions. The Final Streamlit app manages to merge both the models in one diagnostic tool.
The app offers the classification (e.g., Glioma Tumor) and the segmented mask is also presented over the original image. It then counts the number of pixels of the mask to give more of a quantitative size number.

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

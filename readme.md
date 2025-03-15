# Project: Face Mask Detection, Classification, and Segmentation 

## Project submitted by 
-- Abhishek Kumar Singh (MT2024006)
-- Naval Kishore Singh Bisht (MT2024099)

---

This part of project aims to classify images of faces as "with mask" or "without mask" using various machine learning and deep learning techniques.

## Project Structure

```
classification_3.ipynb
dataset/
    with_mask/
        ...
    without_mask/
        ...
```

## Requirements

- Python 3.9
- NumPy
- OpenCV
- scikit-learn
- scikit-image
- TensorFlow

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/aks-master/-VR_Project1_Abhishek_MT2024006.git
    ```

2. Create a new Anaconda environment and install the required packages:
    ```sh
    conda create --name mask_detection_env python=3.9
    conda activate mask_detection_env
    pip install -r requirements.txt
    ```

## Notebook Overview

### Part A: Binary Classification Using Handcrafted Features and ML Classifiers

1. **Feature Extraction**:
    - Extract handcrafted features (HOG) from the dataset.
    - Function: `extract_features(image_path, use_hog)`

2. **Train and Evaluate Classifiers**:
    - Train and evaluate SVM and Neural Network classifiers.
    - Report and compare the accuracy of the classifiers.

### Part B: Binary Classification Using CNN

1. **Data Preparation**:
    - Load the dataset and preprocess images for CNN.
    - One-hot encode labels for categorical cross-entropy loss.

2. **CNN Model 1**:
    - Uses Adam optimizer, deeper layers, and higher dropout.
    - Function: `create_cnn_model_1()`

3. **CNN Model 2**:
    - Uses SGD optimizer, batch normalization, and fewer layers.
    - Function: `create_cnn_model_2()`

4. **Model Training and Evaluation**:
    - Train both CNN models and evaluate their performance.
    - Compare the accuracy of the models.

## Results

## Results

### Comparison of Models:

| Model                      | Accuracy |
|----------------------------|----------|
| Model 1 SVC                | 0.9304   |
| Model 2 MLP neural network | 0.9035   |
| Model 1 (Adam)             | 0.9341   |
| Model 2 (SGD, batch norm)  | 0.7961   |

## Conclusion

The project demonstrates the effectiveness of different machine learning and deep learning techniques for mask classification. CNN Model 1 with Adam optimizer and deeper layers achieved the highest accuracy.

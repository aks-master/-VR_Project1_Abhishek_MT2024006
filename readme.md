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
- tabulate

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

## Dataset

The dataset consists of images categorized into two folders: `with_mask` and `without_mask`. Each folder contains images of faces with and without masks, respectively. The dataset was obtained from [https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)


## Methodology

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

## Hyperparameters and Experiments

###

### SVM 
- sklearn SVC() was used with following hyper parameters (svm_clf.get_params())
```python
{'C': 1.0,
 'break_ties': False,
 'cache_size': 200,
 'class_weight': None,
 'coef0': 0.0,
 'decision_function_shape': 'ovr',
 'degree': 3,
 'gamma': 'scale',
 'kernel': 'rbf',
 'max_iter': -1,
 'probability': False,
 'random_state': None,
 'shrinking': True,
 'tol': 0.001,
 'verbose': False}
```

### MLP neural network
- sklearn MLPClassifier() was used with following hyper parameters
```python
{'activation': 'relu',
 'alpha': 0.0001,
 'batch_size': 'auto',
 'beta_1': 0.9,
 'beta_2': 0.999,
 'early_stopping': False,
 'epsilon': 1e-08,
 'hidden_layer_sizes': (100,),
 'learning_rate': 'constant',
 'learning_rate_init': 0.001,
 'max_fun': 15000,
 'max_iter': 200,
 'momentum': 0.9,
 'n_iter_no_change': 10,
 'nesterovs_momentum': True,
 'power_t': 0.5,
 'random_state': None,
 'shuffle': True,
 'solver': 'adam',
 'tol': 0.0001,
 'validation_fraction': 0.1,
 'verbose': False,
 'warm_start': False}
```

### CNN Model 1
- Optimizer: Adam
- Learning Rate: 0.001
- Dropout: 0.5
- Epochs: 5
- Batch Size: 32

### CNN Model 2
- Optimizer: SGD
- Learning Rate: 0.001
- Momentum: 0.9
- Dropout: 0.3
- Epochs: 5
- Batch Size: 32

## Results

### Comparison of Models:

| Model                      | Accuracy |
|----------------------------|----------|
| Model 1 SVC                | 0.9304   |
| Model 2 MLP neural network | 0.8998   |
| Model 1 (Adam)             | 0.9341   |
| Model 2 (SGD, batch norm)  | 0.7961   |

## Observations and Analysis

- CNN Model 1 with Adam optimizer and higher dropout achieved the highest accuracy of 93.41%.
- CNN models performed better overall because they excel at learning complex patterns, such as spatial and hierarchical features in image data, through their convolutional layers. The ability to extract localized features and combine them in deeper layers gives CNNs a significant advantage, particularly in image classification tasks. In contrast, traditional machine learning models like SVM lack this hierarchical feature learning capability.

## How to Run the Code

1. Ensure you have installed the required packages as mentioned in the Installation section.
2. Open the Jupyter Notebook `classification( Part A B).ipynb`.
3. Run the cells sequentially to execute the code for feature extraction, model training, and evaluation.
4. The results will be displayed in the notebook, including accuracy and confusion matrices for each model.

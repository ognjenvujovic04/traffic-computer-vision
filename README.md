# Traffic Sign Classification with TensorFlow

## Overview
This project utilizes TensorFlow to train a neural network capable of classifying traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The AI model is trained on labeled images and can predict which traffic sign appears in a given photograph. It was developed as part of the CS50’s Introduction to Artificial Intelligence with Python course.

## Installation

1. **Download the distribution code**:  
   ```sh
   wget https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip
   ```
2. **Prepare the dataset**:  
   - Extract the GTSRB dataset.
   - Move the `gtsrb` directory inside your `traffic` project folder.

3. **Install dependencies**:  
   ```sh
   pip3 install -r requirements.txt
   ```
  Required dependencies include:
   - `opencv-python` for image processing  
   - `scikit-learn` for machine learning utilities  
   - `tensorflow` for building and training the neural network  
   - `numpy` for numerical computations  
   - `matplotlib` for data visualization  

## Usage

To train the model and classify traffic signs, run:

```sh
python traffic.py gtsrb
```

To save the trained model:
```sh
python traffic.py gtsrb model.h5
```

## Implementation Details

### 1. **Dataset Structure**
The dataset consists of 43 categories of traffic signs, each represented by a folder containing labeled images. The `load_data` function processes the images, resizes them to `(30,30,3)`, and assigns corresponding labels.

### 2. **Neural Network Model**
The `get_model` function constructs a convolutional neural network (CNN) with the following features:
- Convolutional layers with different filter sizes
- Pooling layers for feature reduction
- Fully connected layers for classification
- Softmax activation in the output layer for multi-class classification

### 3. **Training Process**
- The dataset is split into training and testing sets (60%-40%).
- The model is trained using 10 epochs.
- Performance is evaluated based on accuracy.

### 4. **Performance Results**
Example training output:
```sh
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9564 - loss: 0.1719  
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9517 - loss: 0.1963  
333/333 - 3s - 8ms/step - accuracy: 0.9688 - loss: 0.1256
333/333 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step 
```

### 5. **Misclassified Image Visualization**
The script includes a function to visualize misclassified images, displaying true vs. predicted labels.

## Experimentation & Observations
- Increasing the number of convolutional layers improved accuracy significantly.
- Adding dropout reduced overfitting.
- Using more epochs led to higher accuracy but also increased training time.

## Additional Notes
- The script supports caching for faster data loading.
- Can be adapted for other datasets by modifying the category count dynamically.

## License
This project is part of Harvard's CS50 AI course.

import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from glob import glob

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files, check if they are already cached
    cache_path = "cached_data" + os.sep + sys.argv[1]
    
    if os.path.isdir(cache_path):
        images, labels = load_cached_data(cache_path)
        print("\nData was loaded from cache\n")
    else:
        images, labels = load_data(sys.argv[1])
        save_cached_data(images, labels, cache_path)
        
    if sys.argv[1] != "gtsrb":
        while True:
            try:
                num_categories = int(input("\nInput number of categories for this dataset: "))
                if num_categories>2:
                    break
                else:
                    print("Number should be greater than 1")
            except:
                print("Input number!")
            
    else:
        num_categories = 43

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model(num_categories)

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    start = time.time()
    
    images = []
    labels = []
    
    # for folder in os.listdir(data_dir):
    #     folder_path = data_dir + os.sep + folder
    #     for file in os.listdir(folder_path):
    #         img = cv2.imread(folder_path + os.sep + file)
    #         img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    #         images.append(img)
    #         labels.append(folder)
    
    image_paths = glob(os.path.join(data_dir, '*', '*'))
    
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    
    with Pool(cpu_count()) as pool:
        images = pool.map(load_image, image_paths)
    
    end = time.time()

    print(f"\nTime taken to load the data was {convert_time(end-start)} seconds\n")
    return (images, labels)


def get_model(num_categories):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flattening
        tf.keras.layers.Flatten(),

        # Fully connected layers
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),  

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for sign categories
        tf.keras.layers.Dense(num_categories, activation="softmax")
    ])

    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )   

    return model

def convert_time(seconds):
    """
    Convert seconds into a human-readable format (HH:MM:SS).
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def load_image(file_path):
    """
    Load and resize an image from the given file path.
    """
    img = cv2.imread(file_path)
    if img is not None:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img

def save_cached_data(images, labels, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    np.save(os.path.join(save_path, "images.npy"), images)
    np.save(os.path.join(save_path, "labels.npy"), labels)

def load_cached_data(save_path):
    images = np.load(os.path.join(save_path, "images.npy"))
    labels = np.load(os.path.join(save_path, "labels.npy"))
    return images, labels

if __name__ == "__main__":
    main()

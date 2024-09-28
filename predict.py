import numpy as np  # Matrix processing library
import pandas as pd
import os
from scipy.io import loadmat
from keras.models import load_model

# Set the GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Function to read .mat files from a given folder
def read_mat_files(folder_path):
    """Reads all .mat files from the specified folder."""
    mat_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".mat")]
    data = []
    for file_path in mat_files:
        try:
            mat_data = loadmat(file_path)
            data.append(mat_data)
        except Exception as e:
            print(f"Error reading file: {file_path}")
            print(f"Error details: {e}")
    return data

# Function to read matrix data without labels
def read_values(folder_path, Mat_name):
    """Extracts data from .mat files without labels."""
    mat_data = read_mat_files(folder_path)

    # Print loaded data for debugging
    for data in mat_data:
        print(data)

    size = np.shape(mat_data[0][Mat_name])
    print(size)
    length = len(mat_data)

    # Initialize the data array based on the shape
    if size[0] == 1:
        data_extract = np.zeros([length, size[0], size[1]])
    else:
        data_extract = np.zeros([length, size[0], size[1], size[2]])

    # Extract data from each .mat file
    for i in range(length):
        data_extract[i] = mat_data[i][Mat_name]

    return data_extract

# Function to read matrix data with labels
def read_values_with_labels(folder_path, Mat_name, Label_name):
    """Extracts data and labels from .mat files."""
    mat_data = read_mat_files(folder_path)

    # Print loaded data for debugging
    for data in mat_data:
        print(data)

    size = np.shape(mat_data[0][Mat_name])
    print(size)
    length = len(mat_data)

    # Initialize the data and label arrays
    data_extract = np.zeros([length, size[0], size[1], size[2]])
    label_extract = np.zeros([length])

    for i in range(length):
        data_extract[i] = mat_data[i][Mat_name]
        label_extract[i] = mat_data[i][Label_name]

    return data_extract, label_extract

# Load data and labels from specified folder
Data, label = read_values_with_labels(folder_path="./folder_path/XXX/", Mat_name='D', Label_name='z')


# Select input data based on the desired shape
#Mode
# X1 = X                            # All      9 images
# X1 = X[:, :, :, 0:2]                 # Adjacent 2 images
#X1 = X[:, :, :, [0, 1, 3, 5, 7]]  # Adjacent 5 images
X = Data[:, :, :, [0, 2, 4, 6, 8]]  # Opposite 5 images

# Load the pre-trained model
print("Using loaded model to predict...")
model = load_model("model/XXX.hdf5")


# Make predictions
Y = model.predict(X)

# Prepare results for saving
results = np.zeros([np.shape(Data)[0], 2])
results[:, 0] = Y[:, 0]
results[:, 1] = label * 1e6  # Convert labels to micrometers

# Save results to CSV
pred = pd.DataFrame(results)
pred.to_csv('./pred_results.csv', header=False, index=False)

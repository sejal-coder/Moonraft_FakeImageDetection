# Moonraft_FakeImageDetection

## Overview

This project involves creating a machine learning model to detect image forgery, specifically identifying areas within an image that may have undergone digital modification or have different compression levels. The project uses Error Level Analysis (ELA) and Convolutional Neural Networks (CNNs) to achieve this.

## Setup Instructions

### Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python (3.6 or higher)
- Libraries: Pandas, NumPy, Matplotlib, Pillow (PIL), Scikit-Learn, Keras

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib pillow scikit-learn keras
```

## Running the Code

1. Clone this project repository to your local machine.

2. Open a Jupyter Notebook or Python IDE to run the provided code.

3. Update the file paths and configuration parameters as needed:

   - `TRAIN_DIR`: Path to the folder containing training images.
   - `TEST_DIR`: Path to the folder containing test images.
   - `TEMP_DIR`: Path to a temporary directory for image processing.
   - `ELA_RATE`: Error Level Analysis (ELA) rate used when resaving an image.
   - `IMG_WIDTH` and `IMG_HEIGHT`: Image dimensions for preprocessing.
   - `N_EPOCHS`: Number of training epochs for the CNN.
   - `BATCH_SIZE`: Batch size for training.
   - `DROPOUT_RATE`: Dropout rate for regularization.
   - `SPLIT_RATE`: Splitting proportion for training and validation datasets.

4. Run the code cells in the notebook sequentially. The code includes steps for data processing, model creation, training, and evaluation.

## User Guidelines

### Data

The project assumes a specific folder structure for the training and test datasets (the notebook used a kaggle dataset and was run on kaggle):

- Training images should be organized into subfolders, where each subfolder contains either "original" or "edited" images.
- Test images should be similarly organized into "real" and "edited" subfolders.

### Training the Model

- Ensure that the training and test data paths are correctly configured in the code.
- Run the code cells to preprocess the images, create and train the CNN model.
- The training progress and performance on the validation set will be displayed.

### Model Evaluation

- After training, the code evaluates the model's performance on both the training and validation sets.
- Confusion matrices and classification reports are printed to assess the model's accuracy, precision, recall, and F1-score.

### Testing on Unseen Data

- To test the model on unseen test images, update the file paths for test examples.
- The ELA analysis for test images and model predictions are displayed, along with confusion matrices and classification reports.


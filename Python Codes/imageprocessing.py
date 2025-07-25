import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image # Pillow library for image handling

# --- Configuration ---
# Define paths relative to your project root
DATA_RAW_DIR = 'C:/Skin_Disease_Detection/Skin-Disease-Detection/data/raw'
METADATA_FILE = os.path.join(DATA_RAW_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR_PART1 = os.path.join(DATA_RAW_DIR, 'HAM10000_images_part1')
IMAGE_DIR_PART2 = os.path.join(DATA_RAW_DIR, 'HAM10000_images_part2')

# Target image dimensions for your model
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Number of images to process at once during training

# Diseases to include in your model
# You chose: Melanoma (MEL), Melanocytic Nevus (NV), Basal Cell Carcinoma (BCC), Benign Keratosis-like lesions (BKL)
TARGET_DISEASES = ['mel', 'nv', 'bcc', 'bkl']

# --- Step 1: Load Metadata and Filter Data ---
print("Step 1: Loading metadata and filtering data...")
try:
    df = pd.read_csv(METADATA_FILE)
    print(f"Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_FILE}")
    print("Please ensure 'HAM10000_metadata.csv' is in 'data/raw/'")
    exit()

# Filter for the target diseases
df_filtered = df[df['dx'].isin(TARGET_DISEASES)].copy()
print(f"Filtered dataset shape (only target diseases): {df_filtered.shape}")

# Handle potential missing values (e.g., 'age' can have NaNs)
# For this project, we primarily care about images and diagnoses, but good practice to check.
# df_filtered['age'].fillna(df_filtered['age'].mean(), inplace=True)

# --- Step 2: Map Diagnoses to Numerical Labels ---
# Machine learning models work with numerical labels, not strings.
print("Step 2: Mapping diagnoses to numerical labels...")
label_encoder = LabelEncoder()
df_filtered['label'] = label_encoder.fit_transform(df_filtered['dx'])
num_classes = len(label_encoder.classes_)
print(f"Encoded classes: {list(label_encoder.classes_)}")
print(f"Number of classes: {num_classes}")

# --- Step 3: Create Full Image Paths ---
print("Step 3: Creating full image paths...")
# Create a dictionary to quickly look up image paths
imageid_to_path = {}
for img_id in df_filtered['image_id']:
    # Check in part1 folder
    path1 = os.path.join(IMAGE_DIR_PART1, f"{img_id}.jpg")
    if os.path.exists(path1):
        imageid_to_path[img_id] = path1
    else:
        # Check in part2 folder
        path2 = os.path.join(IMAGE_DIR_PART2, f"{img_id}.jpg")
        if os.path.exists(path2):
            imageid_to_path[img_id] = path2
        else:
            print(f"Warning: Image {img_id}.jpg not found in either part1 or part2 directories.")

# Add the 'path' column to the DataFrame
df_filtered['path'] = df_filtered['image_id'].map(imageid_to_path)

# Drop rows where image path was not found (should be rare if data is complete)
df_filtered.dropna(subset=['path'], inplace=True)
print(f"Dataset shape after path verification: {df_filtered.shape}")

# --- Step 4: Data Splitting ---
# Split data into training, validation, and test sets.
# Stratify ensures that each split has a similar proportion of each disease class.
print("Step 4: Splitting data into train, validation, and test sets...")
X = df_filtered['path']
y = df_filtered['label']

# First split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 75% train, 25% val (from the 80% train+val set)
# This results in 60% train, 20% val, 20% test overall
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# Create DataFrames for each split to use with ImageDataGenerator
train_df = df_filtered[df_filtered['path'].isin(X_train)]
val_df = df_filtered[df_filtered['path'].isin(X_val)]
test_df = df_filtered[df_filtered['path'].isin(X_test)]

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# --- Step 5: Data Augmentation and Generators (On-the-fly processing) ---
# ImageDataGenerator will handle resizing, normalization, and augmentation.
# Rescaling by 1./255 normalizes pixel values from [0, 255] to [0, 1].
print("Step 5: Setting up ImageDataGenerators for on-the-fly processing...")

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values
    rotation_range=20, # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    shear_range=0.2, # Apply shear transformation
    zoom_range=0.2, # Randomly zoom into images
    horizontal_flip=True, # Randomly flip images horizontally
    vertical_flip=True, # Randomly flip images vertically (common for medical images)
    fill_mode='nearest' # Strategy for filling in new pixels created by transformations
)

# Validation and Test data generators (only rescaling, no augmentation)
# Augmentation should ONLY be applied to the training set to avoid data leakage.
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for each split
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path', # Column containing image file paths
    y_col='dx',   # Column containing string labels
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # For multi-class classification
    seed=42 # For reproducibility of augmentation
)

validation_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=42
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Do not shuffle test data to keep order for evaluation metrics
)

print("\nData generators created successfully!")
print("You can now use 'train_generator', 'validation_generator', and 'test_generator' to feed data to your Keras model.")

# --- Optional: Visualize a batch of augmented images ---
print("\nOptional: Visualizing a batch of augmented images from the training generator...")
# Get one batch of images and labels
images, labels = next(train_generator)

# Map numerical labels back to original disease names for display
label_map = {v: k for k, v in train_generator.class_indices.items()}
display_labels = [label_map[np.argmax(label)] for label in labels]

plt.figure(figsize=(10, 10))
for i in range(min(9, len(images))): # Display up to 9 images
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(display_labels[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

print("\nPreprocessing complete. Ready for model building and training!")

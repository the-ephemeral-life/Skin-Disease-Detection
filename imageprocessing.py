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
DATA_RAW_DIR = 'data/raw'
METADATA_FILE = os.path.join(DATA_RAW_DIR, 'HAM10000_metadata.csv')

# IMPORTANT: Adjust these paths based on how you extracted the HAM10000 images.
# You need to find the EXACT folder names where your .jpg images are located.
#
# COMMON SCENARIOS:
# 1. Images are in two folders named 'HAM10000_images_part1' and 'HAM10000_images_part2':
#    IMAGE_DIRS = [os.path.join(DATA_RAW_DIR, 'HAM10000_images_part1'),
#                  os.path.join(DATA_RAW_DIR, 'HAM10000_images_part2')]
#
# 2. Images are in two folders named 'HAM10000_images_part_1' and 'HAM10000_images_part_2' (with underscores):
IMAGE_DIRS = [os.path.join(DATA_RAW_DIR, 'HAM10000_images_part_1'),
              os.path.join(DATA_RAW_DIR, 'HAM10000_images_part_2')]

# Target image dimensions for your model
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Number of images to process at once during training

# Diseases to include in your model
# You chose: Melanoma (MEL), Melanocytic Nevus (NV), Basal Cell Carcinoma (BCC), Benign Keratosis-like lesions (BKL)
TARGET_DISEASES = ['mel', 'nv', 'bcc', 'bkl']

# --- Function to get data generators ---
def get_data_generators(
    metadata_file=METADATA_FILE,
    image_dirs=IMAGE_DIRS,
    target_diseases=TARGET_DISEASES,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE,
    random_state=42 # Added random_state for reproducibility
):
    """
    Loads HAM10000 metadata, filters, creates image paths, splits data,
    and returns TensorFlow Keras ImageDataGenerators for train, validation, and test sets.
    It also returns the training DataFrame for class weight calculation.
    """
    print("Step 1: Loading metadata and filtering data...")
    # Debugging: Print the absolute path the script is looking for
    absolute_metadata_path = os.path.abspath(metadata_file)
    print(f"Attempting to load metadata from: {absolute_metadata_path}")

    try:
        df = pd.read_csv(metadata_file)
        print(f"Original dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file}")
        print(f"Please ensure 'HAM10000_metadata.csv' is in '{os.path.abspath(DATA_RAW_DIR)}'")
        print("Also, ensure you are running the script from your project's root directory.")
        return None, None, None, None # Return None for all if error
    
    # Filter for the target diseases
    df_filtered = df[df['dx'].isin(target_diseases)].copy()
    print(f"Filtered dataset shape (only target diseases): {df_filtered.shape}")

    print("Step 2: Mapping diagnoses to numerical labels...")
    label_encoder = LabelEncoder()
    df_filtered['label'] = label_encoder.fit_transform(df_filtered['dx'])
    num_classes = len(label_encoder.classes_)
    print(f"Encoded classes: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")

    print("Step 3: Creating full image paths...")
    imageid_to_path = {}
    found_images_count = 0
    total_images_to_check = len(df_filtered)

    # Debugging: Print the absolute paths being checked for images
    print(f"Checking for images in directories: {[os.path.abspath(d) for d in image_dirs]}")

    for img_id in df_filtered['image_id']:
        found_path = False
        for img_dir in image_dirs:
            current_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(current_path):
                imageid_to_path[img_id] = current_path
                found_path = True
                found_images_count += 1
                break
        if not found_path:
            # Only print warnings for a few missing images to avoid spamming the console
            if found_images_count < 20 or (total_images_to_check - found_images_count) < 20:
                print(f"Warning: Image {img_id}.jpg not found in any of the specified directories.")

    print(f"Found {found_images_count} out of {total_images_to_check} images.")

    df_filtered['path'] = df_filtered['image_id'].map(imageid_to_path)
    initial_rows = len(df_filtered)
    df_filtered.dropna(subset=['path'], inplace=True)
    dropped_rows = initial_rows - len(df_filtered)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing image paths.")
    print(f"Dataset shape after path verification: {df_filtered.shape}")

    if df_filtered.empty:
        print("\nERROR: No images were found. Please double-check your 'IMAGE_DIRS' configuration.")
        print("Ensure the folder names in 'IMAGE_DIRS' exactly match the folders where your HAM10000 images are stored.")
        print("Example: If your images are in 'data/raw/HAM10000_images_all/', set IMAGE_DIRS = [os.path.join(DATA_RAW_DIR, 'HAM10000_images_all')]")
        return None, None, None, None

    print("Step 4: Splitting data into train, validation, and test sets...")
    X = df_filtered['path']
    y = df_filtered['label']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state, stratify=y_train_val
    )

    train_df = df_filtered[df_filtered['path'].isin(X_train)]
    val_df = df_filtered[df_filtered['path'].isin(X_val)]
    test_df = df_filtered[df_filtered['path'].isin(X_test)]

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    print("Step 5: Setting up ImageDataGenerators for on-the-fly processing...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='dx',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=random_state
    )

    validation_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='dx',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=random_state
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='dx',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("\nData generators created successfully!")
    return train_generator, validation_generator, test_generator, train_df # Also return train_df for class weights


# This part will only run if the script is executed directly (not imported)
if __name__ == "__main__":
    # When run directly, use the module-level constants for visualization
    train_gen, val_gen, test_gen, train_df_for_viz = get_data_generators(
        metadata_file=METADATA_FILE,
        image_dirs=IMAGE_DIRS,
        target_diseases=TARGET_DISEASES,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE
    )

    if train_gen and train_df_for_viz is not None:
        print("\nOptional: Visualizing a batch of augmented images from the training generator...")
        images, labels = next(train_gen)

        label_map = {v: k for k, v in train_gen.class_indices.items()}
        display_labels = [label_map[np.argmax(label)] for label in labels]

        plt.figure(figsize=(10, 10))
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(display_labels[i])
            plt.axis("off")
        plt.tight_layout()
        plt.show()

        print("\nPreprocessing complete. Ready for model building and training!")

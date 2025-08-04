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
AUGMENTED_DIR = "data/augmented"

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
    raw_image_dirs=IMAGE_DIRS,
    augmented_dir=AUGMENTED_DIR,
    target_diseases=TARGET_DISEASES,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE,
    random_state=42 # Added random_state for reproducibility
):
    """
    Loads HAM10000 metadata, filters, creates image paths for both raw and augmented data,
    splits data, and returns TensorFlow Keras ImageDataGenerators for train, validation,
    and test sets. It also returns the training DataFrame for class weight calculation.
    """
    
    # --- Step 1: Load and combine all image data ---
    print("Step 1: Loading and combining raw and augmented image data...")

    # Load original metadata and create a DataFrame for raw images
    try:
        df_raw = pd.read_csv(metadata_file)
        df_filtered = df_raw[df_raw['dx'].isin(target_diseases)].copy()
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file}. Exiting.")
        return None, None, None, None # Return None for all if error

    print(f"Original dataset shape: {df_filtered.shape}")

    # Create image paths for raw images
    imageid_to_path = {}
    found_images_count = 0
    total_images_to_check = len(df_filtered)
    for img_id in df_filtered['image_id']:
        found_path = False
        for img_dir in raw_image_dirs:
            current_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(current_path):
                imageid_to_path[img_id] = current_path
                found_path = True
                found_images_count += 1
                break
        if not found_path and (found_images_count < 20 or (total_images_to_check - found_images_count) < 20):
            print(f"Warning: Image {img_id}.jpg not found in any of the specified raw directories.")

    df_filtered['path'] = df_filtered['image_id'].map(imageid_to_path)
    df_filtered.dropna(subset=['path'], inplace=True)
    print(f"Raw dataset shape after path verification: {df_filtered.shape}")

    # Create a DataFrame for augmented images
    augmented_data = []
    if os.path.exists(augmented_dir):
        print("Found augmented image directory. Scanning for images...")
        for disease in target_diseases:
            disease_path = os.path.join(augmented_dir, disease)
            if os.path.exists(disease_path):
                for file_name in os.listdir(disease_path):
                    if file_name.endswith('.jpg'):
                        augmented_data.append({
                            'path': os.path.join(disease_path, file_name),
                            'dx': disease
                        })
    else:
        print(f"Warning: Augmented directory '{augmented_dir}' not found. Using only raw images.")
    
    if augmented_data:
        df_augmented = pd.DataFrame(augmented_data)
        print(f"Augmented dataset shape: {df_augmented.shape}")
        # Combine the original and augmented DataFrames
        df_combined = pd.concat([df_filtered[['path', 'dx']], df_augmented], ignore_index=True)
    else:
        df_combined = df_filtered[['path', 'dx']]
    
    print(f"Combined dataset shape: {df_combined.shape}")

    if df_combined.empty:
        print("\nERROR: No images were found. Please double-check your directories and file configurations.")
        return None, None, None, None

    # --- Step 2: Map Diagnoses to Numerical Labels ---
    print("\nStep 2: Mapping diagnoses to numerical labels...")
    label_encoder = LabelEncoder()
    df_combined['label'] = label_encoder.fit_transform(df_combined['dx'])
    num_classes = len(label_encoder.classes_)
    print(f"Encoded classes: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")

    # --- Step 3: Data Splitting ---
    print("Step 3: Splitting data into train, validation, and test sets...")
    X = df_combined['path']
    y = df_combined['label']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state, stratify=y_train_val
    )

    train_df = df_combined[df_combined['path'].isin(X_train)]
    val_df = df_combined[df_combined['path'].isin(X_val)]
    test_df = df_combined[df_combined['path'].isin(X_test)]

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # --- Step 4: Data Augmentation and Generators (On-the-fly processing) ---
    print("\nStep 4: Setting up ImageDataGenerators for on-the-fly processing...")
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

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path', # Column containing image file paths
        y_col='dx',   # Column containing string labels
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical', # For multi-class classification
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
    return train_generator, validation_generator, test_generator, train_df



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

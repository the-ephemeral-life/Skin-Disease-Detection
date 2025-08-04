import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image # Pillow library for image handling

# --- Configuration: Setting up file paths and project constants ---
# These variables define where the script looks for files and what settings to use.
# It's a good practice to keep these at the top for easy modification.

# Define paths relative to your project's main folder
DATA_RAW_DIR = 'data/raw'
METADATA_FILE = os.path.join(DATA_RAW_DIR, 'HAM10000_metadata.csv')
AUGMENTED_DIR = 'data/augmented' # NEW: This is the directory for images you've generated

# A list of possible directories where your original images might be stored.
# The script will search in all of them.
# IMPORTANT: Adjust these paths based on how you extracted the HAM10000 images.
# You need to find the EXACT folder names where your .jpg images are located.
IMAGE_DIRS = [os.path.join(DATA_RAW_DIR, 'HAM10000_images_part_1'),
              os.path.join(DATA_RAW_DIR, 'HAM10000_images_part_2')]

# The standard size that your deep learning model expects for input images.
# All images will be resized to this.
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # The number of images to process at once during training

# The specific skin diseases you want your model to learn to identify.
TARGET_DISEASES = ['mel', 'nv', 'bcc', 'bkl']

# --- Function to get data generators: The main logic of the script ---
# This function prepares your data so it can be fed to the machine learning model.
# It's an all-in-one function for data loading, splitting, and preprocessing.
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
    This function loads both original and augmented image data, splits it into
    training, validation, and test sets, and sets up data generators.
    """
    
    # --- Step 1: Load and combine all image data ---
    print("Step 1: Loading and combining raw and augmented image data...")

    # Load the main CSV file which contains labels (diagnoses) for your images.
    try:
        df_raw = pd.read_csv(metadata_file)
        # Keep only the rows that match the diseases you want to work with.
        df_filtered = df_raw[df_raw['dx'].isin(target_diseases)].copy()
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file}. Exiting.")
        return None, None, None, None # Return nothing if the file isn't found

    print(f"Original dataset shape: {df_filtered.shape}")

    # Now, find the actual file path for each original image based on its ID.
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

    # Add the found paths to the DataFrame and remove any entries where the image was missing.
    df_filtered['path'] = df_filtered['image_id'].map(imageid_to_path)
    df_filtered.dropna(subset=['path'], inplace=True)
    print(f"Raw dataset shape after path verification: {df_filtered.shape}")

    # Create a separate list of images you generated in the `augmented_dir`.
    augmented_data = []
    if os.path.exists(augmented_dir):
        print("Found augmented image directory. Scanning for images...")
        # The script assumes your augmented images are organized by disease type.
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
    
    # Combine the original and augmented image data into one large dataset.
    if augmented_data:
        df_augmented = pd.DataFrame(augmented_data)
        print(f"Augmented dataset shape: {df_augmented.shape}")
        # Combine the original and augmented DataFrames
        df_combined = pd.concat([df_filtered[['path', 'dx']], df_augmented], ignore_index=True)
    else:
        # If no augmented data, just use the raw images.
        df_combined = df_filtered[['path', 'dx']]
    
    print(f"Combined dataset shape: {df_combined.shape}")

    if df_combined.empty:
        print("\nERROR: No images were found. Please double-check your directories and file configurations.")
        return None, None, None, None

    # --- Step 2: Map Diagnoses to Numerical Labels ---
    # Machine learning models can't work with text labels like 'mel'.
    # This step converts them into numbers (e.g., mel=0, nv=1).
    print("\nStep 2: Mapping diagnoses to numerical labels...")
    label_encoder = LabelEncoder()
    df_combined['label'] = label_encoder.fit_transform(df_combined['dx'])
    num_classes = len(label_encoder.classes_)
    print(f"Encoded classes: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")

    # --- Step 3: Data Splitting ---
    # This step divides your combined dataset into three distinct groups:
    # - Training set: The largest portion, used to teach the model.
    # - Validation set: Used to check the model's performance during training.
    # - Test set: An unseen set used for final, unbiased evaluation of the model.
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
    # This is a key step! It creates "generators" that feed batches of images
    # to your model in real-time, applying random transformations to the training data.
    print("\nStep 4: Setting up ImageDataGenerators for on-the-fly processing...")
    # The training generator applies many random changes to the images to make the model
    # more robust and prevent it from memorizing the training data.
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values from 0-255 to 0-1
        rotation_range=20, # Rotate images randomly
        width_shift_range=0.2, # Shift images left/right
        height_shift_range=0.2, # Shift images up/down
        shear_range=0.2, # Apply a shearing effect
        zoom_range=0.2, # Zoom in on the image
        horizontal_flip=True, # Flip images horizontally
        vertical_flip=True, # Flip images vertically (useful for medical images)
        fill_mode='nearest' # How to fill in new pixels created by transformations
    )

    # The validation and test generators only perform rescaling, no random changes.
    # This ensures a fair and consistent evaluation of the model.
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # These are the final objects that will be used to train and evaluate the model.
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path', # The column with image file paths
        y_col='dx',   # The column with text labels
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical', # Use this for multi-class classification
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
        shuffle=False # Do not shuffle test data to keep a consistent order for evaluation
    )

    print("\nData generators created successfully!")
    # We return the generators and the training DataFrame.
    return train_generator, validation_generator, test_generator, train_df


# --- Example of how to use the function if this script is run directly ---
# This part of the code is for demonstration and won't run when you import the function.
if __name__ == "__main__":
    train_gen, val_gen, test_gen, train_df_for_viz = get_data_generators()

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

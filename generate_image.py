import os
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img
import cv2 # Used to verify if images are created

# --- Configuration ---
# This script assumes you have your raw images and metadata in the 'data/raw' directory
DATA_RAW_DIR = 'data/raw'
METADATA_FILE = os.path.join(DATA_RAW_DIR, 'HAM10000_metadata.csv')
IMAGE_DIRS = [os.path.join(DATA_RAW_DIR, 'HAM10000_images_part_1'),
              os.path.join(DATA_RAW_DIR, 'HAM10000_images_part_2')]
TARGET_DISEASES = ['mel', 'nv', 'bcc', 'bkl']

# Define the output directory for your newly generated images
OUTPUT_DIR = 'data/augmented'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the number of augmented images to generate per original image
NUM_AUGMENTED_IMAGES_PER_ORIGINAL = 5

# Target image dimensions (should be consistent with your model)
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 1 # We process one image at a time to generate multiple augmented versions

# --- Step 1: Load and filter the metadata ---
print("Step 1: Loading and filtering metadata...")
try:
    df = pd.read_csv(METADATA_FILE)
    df_filtered = df[df['dx'].isin(TARGET_DISEASES)].copy()
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_FILE}. Exiting.")
    exit()

# --- Step 2: Create a temporary directory for ImageDataGenerator to work with ---
# ImageDataGenerator.flow_from_directory() works best with a directory structure
# like 'main_folder/class_1/image.jpg', so we'll create one.
TEMP_DIR = 'temp_aug_input'
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)
for disease in TARGET_DISEASES:
    os.makedirs(os.path.join(TEMP_DIR, disease), exist_ok=True)
print("Step 2: Created temporary directory structure for image generation.")

# Find all image paths and copy them to the temporary directory
image_paths = []
for index, row in df_filtered.iterrows():
    found = False
    for img_dir in IMAGE_DIRS:
        img_path = os.path.join(img_dir, f"{row['image_id']}.jpg")
        if os.path.exists(img_path):
            # Copy the original image to the correct class folder in the temporary directory
            dest_path = os.path.join(TEMP_DIR, row['dx'], f"{row['image_id']}.jpg")
            shutil.copy(img_path, dest_path)
            found = True
            break
    if not found:
        print(f"Warning: Image {row['image_id']} not found.")

# --- Step 3: Configure ImageDataGenerator for augmentation and saving ---
print("Step 3: Configuring ImageDataGenerator...")
datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values
    rotation_range=90, # Randomly rotate images
    width_shift_range=0.6, # Randomly shift images horizontally
    height_shift_range=0.6, # Randomly shift images vertically
    shear_range=0.6, # Apply shear transformation
    zoom_range=0.6, # Randomly zoom into images
    horizontal_flip=True, # Randomly flip images horizontally
    vertical_flip=True, # Randomly flip images vertically
    fill_mode='nearest' # Strategy for filling in new pixels
)

# Use flow_from_directory to read images from the temporary folder
generator = datagen.flow_from_directory(
    TEMP_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None, # We're just generating images, no labels needed
    save_to_dir=OUTPUT_DIR,
    save_prefix='aug', # Prefix for saved augmented images
    save_format='jpg'
)

# --- Step 4: Generate and save the images ---
print(f"Step 4: Generating {NUM_AUGMENTED_IMAGES_PER_ORIGINAL} augmented images for each original image...")
i = 0
total_images_to_generate = len(df_filtered) * NUM_AUGMENTED_IMAGES_PER_ORIGINAL
for batch in generator:
    # This loop will run once for each original image
    i += 1
    if i > len(df_filtered):
        break # Stop after processing all original images once
    
    # We generate multiple augmented versions for each original image
    for _ in range(NUM_AUGMENTED_IMAGES_PER_ORIGINAL):
        # We need to manually generate and save images since flow_from_directory
        # only generates one augmented image per original image in this loop setup.
        # This is a bit of a workaround to get a fixed number of images per original.
        img_array = next(generator)[0] # Get one augmented image
        img = array_to_img(img_array)
        # You can add logic here to save the image with a unique name if needed,
        # but generator.save_to_dir already handles this for us.
        
    print(f"Generated images for {i} out of {len(df_filtered)} original images.")

print("\nImage generation complete!")
print(f"Check the '{OUTPUT_DIR}' directory for the newly generated images.")

# Clean up the temporary directory
shutil.rmtree(TEMP_DIR)
print(f"Cleaned up temporary directory: {TEMP_DIR}")

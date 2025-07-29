import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import numpy as np # Needed for class_weight
import tensorflow.keras.backend as K # Import Keras backend as K to count parameters
import tensorflow.keras.backend as K

# Import the get_data_generators function from your preprocessing script
from imageprocessing import get_data_generators
from sklearn.utils import class_weight # Ensure this is imported for class_weight calculation

# --- Configuration ---
MODEL_SAVE_DIR = 'models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
INITIAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_skin_disease_model.keras')
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'fine_tuned_skin_disease_model.keras')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 4 # Ensure this matches your actual number of classes

# Fine-tuning parameters
FINE_TUNE_EPOCHS = 30 # Number of additional epochs for fine-tuning
# VERY IMPORTANT: Use a much smaller learning rate for fine-tuning
# If performance got worse, this is often the primary culprit.
FINE_TUNE_LEARNING_RATE = 0.000001 # Reduced from 0.00001 to 0.000001

# --- Get Data Generators ---
# Call the function from the preprocessing script to get the generators
print("Getting data generators...")
# Pass the configuration variables to the get_data_generators function
train_generator, validation_generator, test_generator, train_df = get_data_generators(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE
    # Other parameters like metadata_file, image_dirs, target_diseases
    # are already handled by default arguments in get_data_generators
)

# Check if generators were successfully created
if train_generator is None:
    print("ERROR: Data generators could not be created. Exiting.")
    exit()
else:
    print("Data generators (train_generator, validation_generator, test_generator) are available.")


# --- Calculate Class Weights (Highly Recommended for Imbalanced Data) ---
# Use train_df returned by get_data_generators for robust class weight calculation
print("\nCalculating class weights...")
try:
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']), # Use labels from the DataFrame
        y=train_df['label']
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print(f"Calculated class weights: {class_weights_dict}")
except Exception as e:
    print(f"WARNING: Could not calculate class weights: {e}. Skipping class weighting.")
    class_weights_dict = None # Set to None if not using


# --- Step 1: Load the best model from initial training ---
print(f"\nStep 1: Loading the best model from initial training: {INITIAL_MODEL_PATH}")
try:
    model = tf.keras.models.load_model(INITIAL_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load initial model from {INITIAL_MODEL_PATH}. Make sure it exists and was saved correctly.")
    print(f"Error details: {e}")
    exit()

# Print model summary to help identify layer names
print("\nModel Summary after loading (inspect to find base_model layers):")
model.summary()

# --- Step 2: Unfreeze the base model layers ---
print("\nStep 2: Unfreezing specific base model layers for fine-tuning...")

# IMPORTANT CHANGE: Directly identify and set trainable status for layers within the loaded model.
# The ResNet50 base layers are directly part of the 'model' here, not a nested 'base_model' object.

# Find the index of the layer from which to start unfreezing.
# Based on the summary, 'conv5_block1_0_conv' is a good candidate to start unfreezing the last block.
# We'll unfreeze from this layer onwards, including it.
unfreeze_from_layer_name = 'conv5_block1_0_conv'
unfreeze_from_index = -1 # Initialize with -1 to indicate not found

# Iterate through all layers of the loaded model to find the starting point
for i, layer in enumerate(model.layers):
    if layer.name == unfreeze_from_layer_name:
        unfreeze_from_index = i
        break

if unfreeze_from_index == -1:
    print(f"WARNING: Could not find layer '{unfreeze_from_layer_name}'. This might lead to unexpected unfreezing behavior.")
    print("Unfreezing the last 50 layers of the entire model as a fallback.")
    # Fallback: Unfreeze the last N layers of the entire model if the specific layer isn't found.
    # Adjust 50 based on how many layers you want to unfreeze from the end.
    unfreeze_from_index = len(model.layers) - 50
    if unfreeze_from_index < 0:
        unfreeze_from_index = 0 # Ensure index is not negative

# Set trainable status for all layers
for i, layer in enumerate(model.layers):
    if i < unfreeze_from_index:
        layer.trainable = False # Freeze layers before the unfreeze_from_index
    else:
        layer.trainable = True # Unfreeze layers from unfreeze_from_index onwards

print(f"Layers before index {unfreeze_from_index} are frozen. Layers from index {unfreeze_from_index} onwards are unfrozen.")
# Corrected: Use K.count_params
print(f"Total trainable parameters after unfreezing: {np.sum([K.count_params(w) for w in model.trainable_weights])}")


# --- Step 3: Re-compile the model with a low/very low learning rate ---
# IMPORTANT: You MUST re-compile the model after changing `trainable` status.
print("\nStep 3: Re-compiling the model with a very low learning rate for fine-tuning...")
optimizer_fine_tune = Adam(learning_rate=FINE_TUNE_LEARNING_RATE)

model.compile(optimizer=optimizer_fine_tune,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Review the summary to see which layers are now trainable

# --- Step 4: Define Callbacks for Fine-Tuning ---
print("\nStep 4: Defining callbacks for fine-tuning...")
# Use a new checkpoint path for the fine-tuned model
fine_tune_checkpoint_callback = ModelCheckpoint(
    filepath=FINE_TUNED_MODEL_PATH,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# Early stopping and ReduceLROnPlateau can be reused
fine_tune_callbacks = [
    fine_tune_checkpoint_callback,
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000001, verbose=1) # Even lower min_lr
]

# --- Step 5: Continue training (Fine-tuning) ---
print("\nStep 5: Starting model fine-tuning...")
history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=fine_tune_callbacks,
    class_weight=class_weights_dict, # Pass class weights here
    verbose=1
)

print("\nModel fine-tuning complete!")

# --- Step 6: Evaluate the Fine-Tuned Model on the Test Set ---
print("\nStep 6: Evaluating the fine-tuned model on the test set...")
try:
    best_fine_tuned_model = tf.keras.models.load_model(FINE_TUNED_MODEL_PATH)
    print(f"Loaded best fine-tuned model from: {FINE_TUNED_MODEL_PATH}")
    test_loss_fine_tune, test_accuracy_fine_tune = best_fine_tuned_model.evaluate(
        test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1
    )
    print(f"Fine-tuned Test Loss: {test_loss_fine_tune:.4f}")
    print(f"Fine-tuned Test Accuracy: {test_accuracy_fine_tune:.4f}")
except Exception as e:
    print(f"Could not load best fine-tuned model for evaluation: {e}")
    print("Evaluating the last fine-tuned model instead.")
    test_loss_fine_tune, test_accuracy_fine_tune = model.evaluate(
        test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1
    )
    print(f"Fine-tuned Test Loss (last epoch model): {test_loss_fine_tune:.4f}")
    print(f"Fine-tuned Test Accuracy (last epoch model): {test_accuracy_fine_tune:.4f}")


# --- Optional: Plot Training History (Fine-tuning phase) ---
print("\nOptional: Plotting fine-tuning history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_fine_tune.history['accuracy'], label='Train Accuracy (Fine-tune)')
plt.plot(history_fine_tune.history['val_accuracy'], label='Validation Accuracy (Fine-tune)')
plt.title('Model Accuracy (Fine-tuning Phase)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_fine_tune.history['loss'], label='Train Loss (Fine-tune)')
plt.plot(history_fine_tune.history['val_loss'], label='Validation Loss (Fine-tune)')
plt.title('Model Loss (Fine-tuning Phase)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nFine-tuned model saved and evaluated.")
print(f"The fine-tuned model is saved at: {FINE_TUNED_MODEL_PATH}")

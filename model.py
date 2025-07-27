import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import numpy as np # Needed for class_weight

# --- Configuration (Ensure these match your preprocessing script) ---
MODEL_SAVE_DIR = 'models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
INITIAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_skin_disease_model.keras')
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'fine_tuned_skin_disease_model.keras')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 4 

from imageprocessing import get_data_generators

train_generator, validation_generator, test_generator = get_data_generators()
# Fine-tuning parameters
FINE_TUNE_EPOCHS = 30 # Number of additional epochs for fine-tuning
FINE_TUNE_LEARNING_RATE = 0.00001 # VERY IMPORTANT: Use a much smaller learning rate for fine-tuning

# --- IMPORTANT: Ensure your data generators are available ---
# You need to run the 'image-processing-code' script first to get these.
try:
    _ = train_generator
    _ = validation_generator
    _ = test_generator
    print("Data generators (train_generator, validation_generator, test_generator) are available.")
except NameError:
    print("ERROR: Data generators not found. Please ensure 'image-processing-code' script was run successfully.")
    print("Exiting: Data generators are required for model fine-tuning.")
    exit()

# --- Optional: Calculate Class Weights (Highly Recommended for Imbalanced Data) ---
# Assuming train_df is available from the preprocessing script
try:
    from sklearn.utils import class_weight
    # train_df needs to be available from the preprocessing script's scope
    # If not, you'll need to re-load the metadata and split to get train_df
    # For simplicity, assuming train_df is available here.
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.labels), # Use labels from the generator if train_df isn't directly available
        y=train_generator.labels
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print(f"Calculated class weights: {class_weights_dict}")
except NameError:
    print("WARNING: train_generator.labels not available for class weight calculation. Skipping class weighting.")
    print("If you want to use class weights, ensure train_generator.labels is accessible or re-create train_df.")
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

# --- Step 2: Unfreeze the base model layers ---
print("\nStep 2: Unfreezing the base model layers for fine-tuning...")

# Access the base model (ResNet50) within the loaded model
# Assuming the first layer of your model is the ResNet50 base
# You can verify this by checking model.layers[0].name
base_model = model.layers[0]

# Set the entire base model to be trainable
base_model.trainable = True

# --- Step 3: Re-compile the model with a low/very low learning rate ---
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

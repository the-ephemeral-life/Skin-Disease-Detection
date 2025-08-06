import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import numpy as np # Needed for class_weight
import tensorflow.keras.backend as K # Import Keras backend as K to count parameters

# Import the get_data_generators function from your preprocessing script
from imageprocessing import get_data_generators
from sklearn.utils import class_weight # Ensure this is imported for class_weight calculation

# --- Configuration: Setting up file paths and project constants ---
# These variables define where the script looks for files and what settings to use.
# It's a good practice to keep these at the top for easy modification.
MODEL_SAVE_DIR = 'models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Define paths for saving EfficientNet models
# This is the path for the best model from the initial training phase.
INITIAL_EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_efficientnet_model.keras')
# This is a new path for a guaranteed save at the end of Phase 1, in case no 'best' model is saved.
INITIAL_EFFICIENTNET_FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'initial_efficientnet_final_model.keras')
# This is the path for the best model from the fine-tuning phase.
FINE_TUNED_EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'fine_tuned_efficientnet_model.keras')

# The standard size that your deep learning model expects for input images.
# All images will be resized to this.
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # The number of images to process at once during training
NUM_CLASSES = 4 # The number of specific skin diseases you're classifying.

# Training parameters for initial phase (training the new top layers)
INITIAL_EPOCHS = 20 # Can be adjusted, EarlyStopping will manage this automatically.
INITIAL_LEARNING_RATE = 0.001 # A common starting point for EfficientNet, might need tuning.

# Fine-tuning parameters (unfreezing and training the entire model)
FINE_TUNE_EPOCHS = 30 # Number of additional epochs for fine-tuning.
FINE_TUNE_LEARNING_RATE = 0.00001 # A very small learning rate for fine-tuning to prevent "catastrophic forgetting."

# --- Get Data Generators: The main logic of the script ---
# This function call prepares your data so it can be fed to the machine learning model.
# It's an all-in-one function for data loading, splitting, and preprocessing, imported from imageprocessing.py.
print("Getting data generators...")
train_generator, validation_generator, test_generator, train_df = get_data_generators(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE
)

# A safety check to ensure the data was loaded successfully before proceeding.
if train_generator is None:
    print("ERROR: Data generators could not be created. Exiting.")
    exit()
else:
    print("Data generators (train_generator, validation_generator, test_generator) are available.")

# --- Calculate Class Weights: Important for imbalanced datasets ---
# This step helps the model learn from under-represented classes (like melanoma)
# by giving them more importance during training.
print("\nCalculating class weights...")
try:
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_dict = dict(enumerate(class_weights_array))
    print(f"Calculated class weights: {class_weights_dict}")
except Exception as e:
    print(f"WARNING: Could not calculate class weights: {e}. Skipping class weighting.")
    class_weights_dict = None


# --- Phase 1: Initial Training (Frozen EfficientNetB0 Base) ---
# In this phase, we only train the new top layers we added, while keeping the EfficientNet
# base model's weights frozen. This is the first step of transfer learning.
print("\n--- Phase 1: Initial Training (Frozen EfficientNetB0 Base) ---")
print("Step 1: Building the EfficientNetB0 model using transfer learning...")

# Load the EfficientNetB0 model pre-trained on ImageNet.
# 'include_top=False' means we're removing the original classification layers.
base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False,
                                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of the pre-trained base model.
# This prevents their weights from being updated during this phase.
base_model_efficientnet.trainable = False

# Add custom classification layers on top of the base model.
x = base_model_efficientnet.output
x = layers.GlobalAveragePooling2D()(x) # A layer to summarize the features from the base model.
x = layers.Dense(256, activation='relu')(x) # A fully connected layer for learning high-level patterns.
x = layers.Dropout(0.5)(x) # A regularization layer to prevent overfitting.
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x) # The final output layer for classification.

# Create the full model by combining the base and the new top layers.
model_efficientnet = Model(inputs=base_model_efficientnet.input, outputs=predictions)

print("\nModel Summary (Initial EfficientNetB0):")
model_efficientnet.summary()

print("\nStep 2: Compiling the initial model...")
# Configure the model for training with an optimizer, loss function, and metrics.
optimizer_initial = Adam(learning_rate=INITIAL_LEARNING_RATE)
model_efficientnet.compile(optimizer=optimizer_initial,
                           loss='categorical_crossentropy', # Standard for multi-class classification.
                           metrics=['accuracy'])

print("\nStep 3: Defining callbacks for initial training...")
# Callbacks are tools to automatically perform actions during training.
# ModelCheckpoint saves the best model so you don't lose it.
initial_checkpoint_callback = ModelCheckpoint(
    filepath=INITIAL_EFFICIENTNET_MODEL_PATH,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# EarlyStopping stops training if the model's performance on the validation set stops improving.
# This prevents wasting time and overfitting.
# ReduceLROnPlateau reduces the learning rate if the model gets stuck.
initial_callbacks = [
    initial_checkpoint_callback,
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=1)
]

print("\nStep 4: Starting initial model training...")
# The model.fit() function starts the training process.
history_initial = model_efficientnet.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=initial_callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

print("\nInitial model training complete!")
# This guarantees that a model file is saved at the end of the initial phase.
try:
    model_efficientnet.save(INITIAL_EFFICIENTNET_FINAL_MODEL_PATH)
    print(f"Model from final epoch of Phase 1 saved to: {INITIAL_EFFICIENTNET_FINAL_MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Could not save final model from Phase 1. Error: {e}")


# --- Phase 2: Fine-Tuning EfficientNetB0 ---
# In this phase, we unfreeze some of the EfficientNet base layers and retrain the
# entire model with a very low learning rate. This allows the model to fine-tune its
# features specifically for your skin disease dataset.
print("\n--- Phase 2: Fine-Tuning EfficientNetB0 ---")
print("\nStep 5: Loading the best initial model for fine-tuning...")
# We load the best model from the previous phase to start from the best possible state.
try:
    model_to_fine_tune = tf.keras.models.load_model(INITIAL_EFFICIENTNET_MODEL_PATH)
    print(f"Loaded best checkpoint model from: {INITIAL_EFFICIENTNET_MODEL_PATH}")
except:
    print("Best checkpoint model not found. Falling back to final model from initial training.")
    try:
        model_to_fine_tune = tf.keras.models.load_model(INITIAL_EFFICIENTNET_FINAL_MODEL_PATH)
        print(f"Loaded final model from: {INITIAL_EFFICIENTNET_FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Could not load any initial model for fine-tuning. Exiting. Error: {e}")
        exit()


print("\nModel Summary after loading for fine-tuning:")
model_to_fine_tune.summary()


# --- Step 6: Unfreezing specific EfficientNetB0 layers for fine-tuning...
print("\nStep 6: Unfreezing specific EfficientNetB0 layers for fine-tuning...")

# The EfficientNetB0 layers are directly part of model_to_fine_tune.
# We will unfreeze layers from a certain point onwards.
# A good heuristic for EfficientNetB0 is to unfreeze from 'block6a_expand_conv' onwards.
unfreeze_from_layer_name = 'block6a_expand_conv' # Or 'block6b_expand_conv', 'block7a_expand_conv' for more/less unfreezing
unfreeze_start_index = -1

# Find the index of the layer from which to start unfreezing
for i, layer in enumerate(model_to_fine_tune.layers):
    if layer.name == unfreeze_from_layer_name:
        unfreeze_start_index = i
        break

if unfreeze_start_index == -1:
    print(f"WARNING: Could not find specific unfreezing layer '{unfreeze_from_layer_name}'.")
    print("Falling back to unfreezing the last 100 layers of the entire model.")
    # Fallback to unfreeze the last 100 layers of the entire model if the specific block name isn't found
    unfreeze_start_index = len(model_to_fine_tune.layers) - 100
    if unfreeze_start_index < 0:
        unfreeze_start_index = 0

# Set trainable status for all layers in the model
for i, layer in enumerate(model_to_fine_tune.layers):
    if i >= unfreeze_start_index:
        layer.trainable = True # Unfreeze from this point onwards
    else:
        layer.trainable = False # Keep earlier layers frozen

print(f"Layers before index {unfreeze_start_index} are frozen. Layers from index {unfreeze_start_index} onwards are unfrozen.")
# We count and print the total number of trainable parameters to see the effect of unfreezing.
print(f"Total trainable parameters after unfreezing: {np.sum([K.count_params(w) for w in model_to_fine_tune.trainable_weights])}")


print("\nStep 7: Re-compiling the model with a low/very low learning rate for fine-tuning...")
# We must re-compile the model after changing the 'trainable' status of its layers.
optimizer_fine_tune = Adam(learning_rate=FINE_TUNE_LEARNING_RATE)

model_to_fine_tune.compile(optimizer=optimizer_fine_tune,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

print("\nModel Summary (Fine-tuned EfficientNetB0):")
model_to_fine_tune.summary()

print("\nStep 8: Defining callbacks for fine-tuning...")
# We set up new callbacks for the fine-tuning phase.
fine_tune_checkpoint_callback = ModelCheckpoint(
    filepath=FINE_TUNED_EFFICIENTNET_MODEL_PATH,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

fine_tune_callbacks = [
    fine_tune_checkpoint_callback,
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000001, verbose=1)
]

print("\nStep 9: Starting model fine-tuning...")
# We continue training the model from where it left off, but now with the unfrozen layers.
history_fine_tune = model_to_fine_tune.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=fine_tune_callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

print("\nEfficientNetB0 model fine-tuning complete!")

# --- Step 10: Evaluate the Fine-Tuned Model on the Test Set ---
# This is the final and most important evaluation to see how well the model
# generalizes to a completely new dataset.
print("\nStep 10: Evaluating the fine-tuned EfficientNetB0 model on the test set...")
try:
    best_fine_tuned_efficientnet_model = tf.keras.models.load_model(FINE_TUNED_EFFICIENTNET_MODEL_PATH)
    print(f"Loaded best fine-tuned EfficientNetB0 model from: {FINE_TUNED_EFFICIENTNET_MODEL_PATH}")
    test_loss_efficientnet, test_accuracy_efficientnet = best_fine_tuned_efficientnet_model.evaluate(
        test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1
    )
    print(f"Fine-tuned EfficientNetB0 Test Loss: {test_loss_efficientnet:.4f}")
    print(f"Fine-tuned EfficientNetB0 Test Accuracy: {test_accuracy_efficientnet:.4f}")
except Exception as e:
    print(f"Could not load best fine-tuned EfficientNetB0 model for evaluation: {e}")
    print("Evaluating the last trained EfficientNetB0 model instead.")
    test_loss_efficientnet, test_accuracy_efficientnet = model_to_fine_tune.evaluate(
        test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1
    )
    print(f"Fine-tuned EfficientNetB0 Test Loss (last epoch model): {test_loss_efficientnet:.4f}")
    print(f"Fine-tuned EfficientNetB0 Test Accuracy (last epoch model): {test_accuracy_efficientnet:.4f}")


# --- Optional: Plot Training History (Fine-tuning phase) ---
# We visualize the training history to see how the model's accuracy and loss changed over time.
# This helps in diagnosing issues like overfitting or underfitting.
print("\nOptional: Plotting EfficientNetB0 training history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_initial.history['accuracy'], label='Initial Train Accuracy')
plt.plot(history_initial.history['val_accuracy'], label='Initial Val Accuracy')
plt.plot(history_fine_tune.history['accuracy'], label='Fine-tune Train Accuracy')
plt.plot(history_fine_tune.history['val_accuracy'], label='Fine-tune Val Accuracy')
plt.title('EfficientNetB0 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_initial.history['loss'], label='Initial Train Loss')
plt.plot(history_initial.history['val_loss'], label='Initial Val Loss')
plt.plot(history_fine_tune.history['loss'], label='Fine-tune Train Loss')
plt.plot(history_fine_tune.history['val_loss'], label='Fine-tune Val Loss')
plt.title('EfficientNetB0 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nEfficientNetB0 model training and fine-tuning complete.")
print(f"The best fine-tuned EfficientNetB0 model is saved at: {FINE_TUNED_EFFICIENTNET_MODEL_PATH}")

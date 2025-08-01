import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import numpy as np # Needed for class_weight

# Import the get_data_generators function from your preprocessing script
from imageprocessing import get_data_generators
from sklearn.utils import class_weight # Ensure this is imported for class_weight calculation

# --- Configuration ---
MODEL_SAVE_DIR = 'models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# Define paths for saving EfficientNet models
INITIAL_EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_efficientnet_model.keras')
FINE_TUNED_EFFICIENTNET_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'fine_tuned_efficientnet_model.keras')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 4 # Ensure this matches your actual number of classes

# Training parameters for initial phase
INITIAL_EPOCHS = 20 # Can be adjusted, EarlyStopping will manage
INITIAL_LEARNING_RATE = 0.001 # A common starting point for EfficientNet, might need tuning

# Fine-tuning parameters
FINE_TUNE_EPOCHS = 30 # Number of additional epochs for fine-tuning
FINE_TUNE_LEARNING_RATE = 0.00001 # A very small learning rate for fine-tuning

# --- Get Data Generators ---
print("Getting data generators...")
train_generator, validation_generator, test_generator, train_df = get_data_generators(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE
)

if train_generator is None:
    print("ERROR: Data generators could not be created. Exiting.")
    exit()
else:
    print("Data generators (train_generator, validation_generator, test_generator) are available.")

# --- Calculate Class Weights (Highly Recommended for Imbalanced Data) ---
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
print("\n--- Phase 1: Initial Training (Frozen EfficientNetB0 Base) ---")
print("Step 1: Building the EfficientNetB0 model using transfer learning...")

base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False,
                                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

base_model_efficientnet.trainable = False # Freeze the base model

x = base_model_efficientnet.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model_efficientnet = Model(inputs=base_model_efficientnet.input, outputs=predictions)

print("\nModel Summary (Initial EfficientNetB0):")
model_efficientnet.summary()

print("\nStep 2: Compiling the initial model...")
optimizer_initial = Adam(learning_rate=INITIAL_LEARNING_RATE)
model_efficientnet.compile(optimizer=optimizer_initial,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

print("\nStep 3: Defining callbacks for initial training...")
initial_checkpoint_callback = ModelCheckpoint(
    filepath=INITIAL_EFFICIENTNET_MODEL_PATH,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

initial_callbacks = [
    initial_checkpoint_callback,
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=1)
]

print("\nStep 4: Starting initial model training...")
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

# --- Phase 2: Fine-Tuning EfficientNetB0 ---
print("\n--- Phase 2: Fine-Tuning EfficientNetB0 ---")
print(f"\nStep 5: Loading the best initial model for fine-tuning: {INITIAL_EFFICIENTNET_MODEL_PATH}")
try:
    # Load the best model from the initial training phase
    model_efficientnet = tf.keras.models.load_model(INITIAL_EFFICIENTNET_MODEL_PATH)
    print("Initial best model loaded successfully for fine-tuning.")
except Exception as e:
    print(f"ERROR: Could not load initial EfficientNet model from {INITIAL_EFFICIENTNET_MODEL_PATH}. Make sure it exists and was saved correctly.")
    print(f"Error details: {e}")
    exit()

print("\nStep 6: Unfreezing specific EfficientNetB0 layers for fine-tuning...")

# Find the EfficientNetB0 base model within the loaded model
efficientnet_base = None
for layer in model_efficientnet.layers:
    if isinstance(layer, Model) and 'efficientnetb0' in layer.name: # EfficientNetB0 layer name
        efficientnet_base = layer
        break

if efficientnet_base is None:
    print("ERROR: Could not find the EfficientNetB0 base model within the loaded model. Exiting.")
    print("Please inspect model_efficientnet.summary() output carefully to identify the EfficientNetB0 base layer's name.")
    exit()
else:
    print(f"Found EfficientNetB0 base model: {efficientnet_base.name}")

efficientnet_base.trainable = True # Set the base model to be trainable

# IMPORTANT: Implement partial unfreezing.
# EfficientNet models have a different layer structure than ResNet.
# You typically unfreeze a certain percentage or from a specific block.
# For EfficientNetB0, unfreezing the last ~30-50% of layers is a good starting point.
# Let's unfreeze layers from a certain percentage point.
# EfficientNetB0 has around 230-240 layers (including activations, BN etc. within blocks)
# We'll unfreeze the last ~100 layers as a starting point.
unfreeze_from_index = len(efficientnet_base.layers) - 100 # Unfreeze last 100 layers
if unfreeze_from_index < 0:
    unfreeze_from_index = 0 # Ensure index is not negative

for layer in efficientnet_base.layers[:unfreeze_from_index]:
    layer.trainable = False

print(f"Kept {unfreeze_from_index} layers of the EfficientNetB0 base model frozen. Unfrozen {len(efficientnet_base.layers) - unfreeze_from_index} layers.")
print(f"Total trainable parameters after unfreezing: {np.sum([K.count_params(w) for w in model_efficientnet.trainable_weights])}")


print("\nStep 7: Re-compiling the model with a very low learning rate for fine-tuning...")
optimizer_fine_tune = Adam(learning_rate=FINE_TUNE_LEARNING_RATE)

model_efficientnet.compile(optimizer=optimizer_fine_tune,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

print("\nModel Summary (Fine-tuned EfficientNetB0):")
model_efficientnet.summary() # Review the summary to see which layers are now trainable

print("\nStep 8: Defining callbacks for fine-tuning...")
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
history_fine_tune = model_efficientnet.fit(
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
    test_loss_efficientnet, test_accuracy_efficientnet = model_efficientnet.evaluate(
        test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1
    )
    print(f"Fine-tuned EfficientNetB0 Test Loss (last epoch model): {test_loss_efficientnet:.4f}")
    print(f"Fine-tuned EfficientNetB0 Test Accuracy (last epoch model): {test_accuracy_efficientnet:.4f}")


# --- Optional: Plot Training History (Fine-tuning phase) ---
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

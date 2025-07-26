import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt # For plotting training history

# --- Configuration (Ensure these match your preprocessing script) ---
# Define paths for saving the model
MODEL_SAVE_DIR = 'models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create models directory if it doesn't exist

# Target image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Number of classes for specific problem (MEL, NV, BCC, BKL = 4 classes)
# This should match the number of unique 'dx' values you filtered for.
# You can get this from train_generator.num_classes or len(label_encoder.classes_)
NUM_CLASSES = 4 # Adjust if you include more diseases later

# Training parameters
EPOCHS = 50 # You might need more or less, EarlyStopping will help
LEARNING_RATE = 0.0001 # Start with a small learning rate for transfer learning

# --- IMPORTANT: Ensure your data generators are available ---
# You need to run the 'image-processing-code' script first to get these.
from imageprocessing import get_data_generators
train_generator, validation_generator, test_generator = get_data_generators()

# Placeholder for generators if running this script standalone for testing model definition
# In a real scenario, these would be populated by your image processing script.
try:
    # Attempt to access the generators from the previous script's execution
    _ = train_generator
    _ = validation_generator
    _ = test_generator
    print("Data generators (train_generator, validation_generator, test_generator) are available.")
except NameError:
    print("WARNING: Data generators not found. Please ensure 'image-processing-code' script was run successfully.")
    print("This script will define the model, but cannot train without the generators.")
    # If generators are not available, we cannot proceed with training, so exit or mock them.
    # For now, we'll just print a warning.
    # In a production script, you might raise an error or call the preprocessing function here.
    exit("Exiting: Data generators are required for model training.")


# --- Step 1: Build the ResNet50 Model using Transfer Learning ---
print("Step 1: Building the ResNet50 model using transfer learning...")

# Load the ResNet50 model pre-trained on ImageNet, excluding the top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of the pre-trained base model
# This prevents their weights from being updated during the initial training phase,
# preserving the learned features from ImageNet.
base_model.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = layers.GlobalAveragePooling2D()(x) # Reduces spatial dimensions, averages features
x = layers.Dense(256, activation='relu')(x) # A dense layer with ReLU activation
x = layers.Dropout(0.5)(x) # Dropout for regularization to prevent overfitting
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x) # Output layer with softmax for multi-class classification

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

model.summary() # Print a summary of the model architecture

# --- Step 2: Compile the Model ---
print("\nStep 2: Compiling the model...")
# Use Adam optimizer with a specific learning rate
optimizer = Adam(learning_rate=LEARNING_RATE)

# For multi-class classification with one-hot encoded labels (from class_mode='categorical')
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Step 3: Define Callbacks ---
print("\nStep 3: Defining callbacks for training...")
# ModelCheckpoint: Saves the best model weights based on validation accuracy
checkpoint_filepath = os.path.join(MODEL_SAVE_DIR, 'best_skin_disease_model.keras') # .keras is the recommended format
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False, # Save the entire model
    monitor='val_accuracy', # Monitor validation accuracy
    mode='max', # Save when validation accuracy is maximized
    save_best_only=True, # Only save the best model
    verbose=1
)

# EarlyStopping: Stops training if validation loss doesn't improve for 'patience' epochs
early_stopping_callback = EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=10, # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity
    verbose=1
)

# ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving
reduce_lr_on_plateau_callback = ReduceLROnPlateau(
    monitor='val_loss', # Monitor validation loss
    factor=0.1, # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=5, # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=0.000001, # Lower bound on the learning rate
    verbose=1
)

callbacks = [
    model_checkpoint_callback,
    early_stopping_callback,
    reduce_lr_on_plateau_callback
]

# --- Step 4: Train the Model ---
print("\nStep 4: Starting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Number of batches per epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE, # Number of validation batches
    callbacks=callbacks,
    verbose=1
)

print("\nModel training complete!")

# --- Step 5: Evaluate the Model on the Test Set ---
print("\nStep 5: Evaluating the model on the test set...")
# Load the best model saved by ModelCheckpoint for final evaluation
# This ensures you evaluate the model that performed best on the validation set.
try:
    best_model = tf.keras.models.load_model(checkpoint_filepath)
    print(f"Loaded best model from: {checkpoint_filepath}")
    test_loss, test_accuracy = best_model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
except Exception as e:
    print(f"Could not load best model for evaluation: {e}")
    print("Evaluating the last trained model instead.")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE, verbose=1)
    print(f"Test Loss (last epoch model): {test_loss:.4f}")
    print(f"Test Accuracy (last epoch model): {test_accuracy:.4f}")


# --- Optional: Plot Training History ---
print("\nOptional: Plotting training history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nModel building and initial training complete. The best model is saved.")
print(f"The trained model is saved at: {checkpoint_filepath}")

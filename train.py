import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 1. SETTINGS & PATHS
DATASET_DIR = "dataset"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # Standard size for image classification models

print("Loading dataset...")

# 2. LOAD TRAINING DATA
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 3. LOAD VALIDATION DATA
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get the class names (should print: ['100', '1000', '200', '2000', '50', '500'])
class_names = train_ds.class_names
print(f"Detected Classes: {class_names}")

# Optimize data loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. DEFINE DATA AUGMENTATION
# This randomly flips, rotates, and zooms the images during training to help the 1000 DA class!
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
])

# 5. BUILD THE CNN MODEL
num_classes = len(class_names)

model = models.Sequential([
    # Data Augmentation layer
    data_augmentation,
    
    # Rescale pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the 3D features into 1D vectors
    layers.Flatten(),
    
    # Dense (Fully Connected) Hidden Layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevents overfitting
    
    # Output Layer (one neuron per currency class)
    layers.Dense(num_classes)
])

# 6. COMPILE THE MODEL
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# 7. TRAIN THE MODEL
EPOCHS = 15
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 8. SAVE THE MODEL FOR FUTURE USE
model.save("algerian_money_model.keras")
print("Model saved successfully as 'algerian_money_model.keras'!")

# 9. PLOT TRAINING RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
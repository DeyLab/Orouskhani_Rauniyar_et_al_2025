import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set paths for the Microscopy images (folders: 'CS', 'CS_with_Cholic_acid')
train_dir = 'training'  # Training data path
test_dir = 'testing'    # Testing data path

# ImageDataGenerator for loading and augmenting images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training images (80% of the data)
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Load validation images (20% of the data)
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load test images (no validation split)
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load the DenseNet base model pre-trained on ImageNet, excluding its top layers
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the DenseNet base model to prevent its weights from being updated during training
base_model.trainable = False

# Build the custom classification head
x = base_model.output
x = Flatten()(x)                     # Flatten the 4D tensor to 1D
x = Dense(128, activation='relu')(x)   # Fully connected layer with 128 neurons and ReLU activation
predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

# Create the final model by combining the DenseNet base and the custom head
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with the Adam optimizer (default beta parameters) and a learning rate of 0.001
model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=20,  # Adjust the number of epochs as needed
    validation_data=val_data
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy}")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy}")

# Plot training and validation accuracy and loss
epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

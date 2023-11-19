import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load the annotations into a DataFrame
train_annotations = pd.read_csv('American Sign Language Letters.v1-v1.tensorflow/train/_annotations.csv')
valid_annotations = pd.read_csv('American Sign Language Letters.v1-v1.tensorflow/valid/_annotations.csv')

# Calculate class weights to counter class imbalances
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_annotations['class']),
    y=train_annotations['class'].values
)
class_weights = dict(enumerate(class_weights))

# Image preprocessing function
def preprocess_image(image):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float values in [0, 1]
    return tf.image.resize(image, [200, 200])

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for the validation set
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_annotations,
    directory='American Sign Language Letters.v1-v1.tensorflow/train',
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_annotations,
    directory='American Sign Language Letters.v1-v1.tensorflow/valid',
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

# Load the VGG16 model, pre-trained on ImageNet
base_model = VGG16(input_shape=(200, 200, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the VGG16 model

# Add custom layers on top of VGG16
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('asl_alphabet_model_vgg16', save_best_only=True, verbose=1, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save the final model in TensorFlow SavedModel format
model.save('final_asl_alphabet_model')

# Replace 'path_to_your_train_annotations.csv', 'path_to_your_valid_annotations.csv', 
# 'path_to_your_training_images', and 'path_to_your_validation_images' with the actual paths to your files.

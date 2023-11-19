import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the annotations into a DataFrame
annotations = pd.read_csv('American Sign Language Letters.v1-v1.tensorflow/train/_annotations.csv')
# Assuming your CSV has 'filename' and 'class' columns
# Adjust column names if necessary

# Split data into training and validation sets
train_df, valid_df = train_test_split(annotations, test_size=0.2, stratify=annotations['class'])

# Set up the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create generators
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='American Sign Language Letters.v1-v1.tensorflow/train',  # Update this to the path of your images
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory='American Sign Language Letters.v1-v1.tensorflow/train',  # Update this to the path of your images
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=25
)

# Save the model
model.save('asl_alphabet_model.h5')

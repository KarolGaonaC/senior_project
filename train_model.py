import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model_definitions import build_resnet, build_nasnet, build_inceptionv3, build_merged_model
import matplotlib.pyplot as plt

# define the desired image dimensions
img_width, img_height = 224, 224  # Input image dimensions expected by the models
epochs = 10
batch_size = 32
num_classes = 120

# data directories
train_data_dir = '/cluster/home/kigaona20/seniorProject/train_data'
test_data_dir = '/cluster/home/kigaona20/seniorProject/test_data'

# cata generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

print("Train Data Generator Configuration:")
print("Rescaling Factor:", train_datagen.rescale)
print("Horizontal Flip:", train_datagen.horizontal_flip)
# Add more print statements for other parameters as needed

print("\nTest Data Generator Configuration:")
print("Rescaling Factor:", test_datagen.rescale)
# Add more print statements for other parameters as needed



# create individual models
resnet_model = build_resnet((img_width, img_height, 3), num_classes=num_classes)
nasnet_model = build_nasnet((img_width, img_height, 3), num_classes=num_classes)
inceptionv3_model = build_inceptionv3((img_width, img_height, 3), num_classes=num_classes)

# combine the generators
train_generator_resnet = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

train_generator_nasnet = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

train_generator_inceptionv3 = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# after defining individual models, extract their outputs
resnet_output = resnet_model.layers[-1].output
nasnet_output = nasnet_model.layers[-1].output
inceptionv3_output = inceptionv3_model.layers[-1].output

# pass the outputs to build_merged_model function
merged_model = build_merged_model(resnet_model, nasnet_model, inceptionv3_model, num_classes)

print("Number of input tensors expected by the model:", len(merged_model.inputs))

# compile model
merged_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Calculate steps_per_epoch based on the generator that yields the maximum number of batches
steps_per_epoch = max(len(train_generator_resnet), len(train_generator_nasnet), len(train_generator_inceptionv3))

# Obtener los datos de cada generador
data_resnet = train_generator_resnet.next()
data_nasnet = train_generator_nasnet.next()
data_inceptionv3 = train_generator_inceptionv3.next()

# Print the input shapes before training
print("Input shapes:")
print("ResNet input shape:", data_resnet[0].shape)
print("NASNet input shape:", data_nasnet[0].shape)
print("InceptionV3 input shape:", data_inceptionv3[0].shape)

def custom_data_generator(train_datagen, train_data_dir, img_width, img_height, batch_size):
    # Initialize data generators
    resnet_data = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    nasnet_data = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    inceptionv3_data = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    while True:
        # Yield inputs for the merged model and their corresponding labels
        resnet_images, resnet_labels = resnet_data.next()
        nasnet_images, nasnet_labels = nasnet_data.next()
        inceptionv3_images, inceptionv3_labels = inceptionv3_data.next()
        
        # Yield inputs for the merged model and their corresponding labels
        yield [resnet_images, nasnet_images, inceptionv3_images], resnet_labels

# Create custom data generator
train_generator = custom_data_generator(train_datagen, train_data_dir, img_width, img_height, batch_size)

print("Starting model training...")
# Train the model using the custom data generator
history = merged_model.fit(
    x=train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
print("Model training completed.")

# print out history
print("Training Loss:", history.history['loss'])
print("Training Accuracy:", history.history['accuracy'])
print("Validation Loss:", history.history['val_loss'])
print("Validation Accuracy:", history.history['val_accuracy'])

# Inside the loop for testing each batch
for i, (x_batch, y_batch) in enumerate(validation_generator):
    print(f"Testing batch {i+1}/{len(validation_generator)}")
    # Add more print statements here to inspect the input batch, if needed

    # Test the model on the current batch
    loss, accuracy = merged_model.evaluate(x_batch, y_batch, verbose=0)
    print(f"Batch {i+1} - Loss: {loss}, Accuracy: {accuracy}")

# After testing all batches
print("Model testing completed.")

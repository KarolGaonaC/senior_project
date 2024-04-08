from tensorflow.keras.applications import ResNet50, NASNetLarge, InceptionV3
from tensorflow.keras.layers import Concatenate, Dense, GlobalAveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model

def build_resnet(input_shape, num_classes):
    # Load ResNet50 model with pre-trained weights
    base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the weights of the pre-trained layers
    for layer in base_resnet.layers:
        layer.trainable = False
    
    # Add custom layers on top of the pre-trained layers
    x = base_resnet.output
    print("Shape after base ResNet output:", x.shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    print("Shape after Conv2D layer 1:", x.shape)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    print("Shape after MaxPooling2D layer 1:", x.shape)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    print("Shape after Conv2D layer 2:", x.shape)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    print("Shape after MaxPooling2D layer 2:", x.shape)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    print("Shape after Conv2D layer 3:", x.shape)
    x = GlobalAveragePooling2D()(x)
    print("Shape after GlobalAveragePooling2D layer:", x.shape)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create a new model combining the base ResNet model and custom layers
    model = Model(inputs=base_resnet.input, outputs=predictions)
    
    return model


def build_nasnet(input_shape, num_classes):
    base_nasnet = NASNetLarge(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_nasnet.output
    print("Shape after base NASNetLarge output:", x.shape)
    x = GlobalAveragePooling2D()(x)
    print("Shape after GlobalAveragePooling2D layer:", x.shape)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_nasnet.input, outputs=predictions)
    return model

def build_inceptionv3(input_shape, num_classes):
    base_inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_inceptionv3.output
    print("Shape after base InceptionV3 output:", x.shape)
    x = GlobalAveragePooling2D()(x)
    print("Shape after GlobalAveragePooling2D layer:", x.shape)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_inceptionv3.input, outputs=predictions)
    return model

def build_merged_model(resnet_model, nasnet_model, inceptionv3_model, num_classes):
    # Get the output tensors of the individual models
    print("Building merged model...")
    resnet_output = resnet_model.output
    nasnet_output = nasnet_model.output
    inceptionv3_output = inceptionv3_model.output

    # Print the shapes of the output tensors
    print("ResNet output shape:", resnet_output.shape)
    print("NASNet output shape:", nasnet_output.shape)
    print("InceptionV3 output shape:", inceptionv3_output.shape)

    # Concatenate the output tensors of the individual models
    merged_features = Concatenate()([resnet_output, nasnet_output, inceptionv3_output])
    
    # Additional layers before final classification
    x = Dense(512, activation='relu')(merged_features)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the merged model
    merged_model = Model(inputs=[resnet_model.input, nasnet_model.input, inceptionv3_model.input], outputs=predictions)
    print("Merged model built successfully.")
    return merged_model

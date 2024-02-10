from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dropout, MaxPooling2D, Flatten, \
    Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense


# Define the model
def build_model3():
    inputs = Input(shape=(32, 32, 3))

    # First Convolutional Block with shortcut connection
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)  # Adding dropout after activation
    shortcut = Conv2D(32, (1, 1), strides=(2, 2), padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])

    shortcut = x  # Save input for shortcut
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    shortcut = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])

    shortcut = x  # Save input for shortcut
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    shortcut = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])  # Add shortcut connection

    for _ in range(4):
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

    # MaxPooling
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    # Flatten and De
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    # Creating the model
    model = Model(inputs=inputs, outputs=outputs, name='model3_with_shortcuts')
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


model = build_model3()


# Function to calculate the output size of each layer
def get_output_size(model, layer_index):
    # Use a temporary model to get the output shape of a specific layer
    temp_model = Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    return temp_model.output_shape[1:]


# Function to calculate MACs (Multiply-Accumulate Operations) for each layer
def calculate_macs(layer):

    if 'Conv2D' in layer.__class__.__name__:
        output_shape = layer.output_shape[1:]  # Exclude the batch size
        kernel_size = layer.kernel_size
        filters = layer.filters
        macs_per_output_element = kernel_size[0] * kernel_size[1] * output_shape[-1]
        total_macs = filters * macs_per_output_element * output_shape[0] * output_shape[1]
        return total_macs

    elif 'Dense' in layer.__class__.__name__:
        units = layer.units
        input_shape = layer.input_shape
        total_macs = units * input_shape[1]
        return total_macs
    else:
        return 0

layers_summary = []
for i, layer in enumerate(model.layers):
    layer_type = layer.__class__.__name__
    filters_units = getattr(layer, 'filters', getattr(layer, 'units', '-'))
    parameters = layer.count_params()
    macs = calculate_macs(layer)
    output_size = get_output_size(model, i)

    layer_summary = {
        'Layer Type': layer_type,
        'Filters/Units': filters_units,
        'Parameters': parameters,
        'MACs': macs,
        'Output Size': output_size
    }
    layers_summary.append(layer_summary)

# Print the layers summary
for layer_info in layers_summary:
    print(layer_info)



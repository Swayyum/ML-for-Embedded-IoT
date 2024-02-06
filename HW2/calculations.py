# Constants for the CNN layers
input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 channels
kernel_size = 3  # 3x3 kernel size
strides_conv = [2, 2, 2]  # Strides for the first three conv layers
filters = [32, 64, 128]  # Filters for the conv layers
dense_units = [128, 10]  # Units in the dense layers

# Calculations
layer_details = []
output_shape = input_shape

def calc_conv2d_params_and_macs(input_shape, num_filters, kernel_size, stride, padding):
    # Output size calculation for "same" padding
    output_height = (input_shape[0] - 1) // stride + 1
    output_width = (input_shape[1] - 1) // stride + 1
    output_shape = (output_height, output_width, num_filters)

    # Parameters calculation
    num_params = (input_shape[2] * kernel_size * kernel_size + 1) * num_filters

    # MACs calculation
    macs = output_height * output_width * num_filters * (input_shape[2] * kernel_size * kernel_size)

    return num_params, macs, output_shape

# Calculate for Conv2D and BatchNorm layers
for idx, num_filters in enumerate(filters):
    stride = strides_conv[idx]
    num_params, macs, output_shape = calc_conv2d_params_and_macs(output_shape, num_filters, kernel_size, stride, "same")
    layer_details.append((f"Conv2D-{num_filters}", num_params, macs, output_shape))

    # BatchNorm parameters and MACs (simplified)
    batchnorm_params = 2 * num_filters  # scale and shift parameters
    layer_details.append((f"BatchNorm-{num_filters}", batchnorm_params, 0, output_shape))  # MACs considered negligible

# Example calculation for subsequent layers without stride
subsequent_filters = 128
for _ in range(4):  # Four more pairs of Conv2D+BatchNorm without stride
    num_params, macs, output_shape = calc_conv2d_params_and_macs(output_shape, subsequent_filters, kernel_size, 1, "same")
    layer_details.append((f"Conv2D-{subsequent_filters}", num_params, macs, output_shape))
    batchnorm_params = 2 * subsequent_filters
    layer_details.append((f"BatchNorm-{subsequent_filters}", batchnorm_params, 0, output_shape))

# MaxPooling, Flatten, and Dense Layers calculation placeholder
# Placeholder for the output of the last convolutional layer before max pooling
last_conv_output_shape = output_shape

# MaxPooling
pool_size = 4
stride_pool = 4
output_height = (last_conv_output_shape[0] - 1) // stride_pool + 1
output_width = (last_conv_output_shape[1] - 1) // stride_pool + 1
output_shape = (output_height, output_width, last_conv_output_shape[2])
layer_details.append((f"MaxPooling", 0, 0, output_shape))

# Flatten
flatten_size = output_shape[0] * output_shape[1] * output_shape[2]
layer_details.append((f"Flatten", 0, 0, (flatten_size,)))

# Dense layer calculations
for units in dense_units:
    dense_params = (flatten_size if units == dense_units[0] else dense_units[0]) * units + units
    dense_macs = dense_params - units  # Subtract bias terms for MAC calculation
    layer_details.append((f"Dense-{units}", dense_params, dense_macs, (units,)))
    flatten_size = units  # Update for the next layer if any

layer_details
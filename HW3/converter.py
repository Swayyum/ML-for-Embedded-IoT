import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
import os


model = load_model(r'C:/Users/X390 Yoga/Desktop/Swayam/ML IoT/HW3/hw3-spr2024-Swayyum-main/hw3-spr2024-Swayyum-main/src')


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Define the number of calibration steps
num_calibration_steps = 100

# Define a representative dataset generator function
def representative_dataset_gen():
    # Yield a representative set of input tensors
    for _ in range(num_calibration_steps):
        # Generate example input data (replace this with actual input data)
        input_shape = (1, 28, 28)  # Example shape, adjust as per your model input shape
        input_data = tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)
        yield [input_data]

# Set the representative dataset generator function
converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()

# Step 3: Export the TensorFlow Lite model as a C byte array using xxd -i
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# import os
# os.system("xxd -i model_quantized.tflite > model_quantized_data.cc")
    
# def convert_to_c_array(byte_data, var_name="model_data"):
#     """Convert binary data to a C array string."""
#     array_content = ", ".join(f"0x{b:02x}" for b in byte_data)
#     return f"const unsigned char {var_name}[] = {{\n  {array_content}\n}};\nunsigned int {var_name}_len = {len(byte_data)};"

# # Read the TFLite model's binary data
# with open('model_quantized.tflite', 'rb') as model_file:
#     tflite_model_data = model_file.read()

# # Convert the model's binary data to a C array string
# c_array_str = convert_to_c_array(tflite_model_data)

# # Optionally, write the C array string to a file
# with open('model_quantized_data.cc', 'w') as c_file:
#     c_file.write(c_array_str)

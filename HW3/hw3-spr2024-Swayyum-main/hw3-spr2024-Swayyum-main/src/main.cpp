#include <Arduino.h>
#include <TensorFlowLite.h>
#include <sin.h>

#include <wire.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "sin_predictor.cc.cpp"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

 // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddSub();
  micro_op_resolver.AddMean();

    // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

void loop() {
  if (chars_avail > 0) {
    received_char = Serial.read();
    Serial.print(received_char);

    in_str_buff[in_buff_idx++] = received_char;
    
if (received_char ==13) {
  Serial.print("About to process line: ");
  Serial.println(in_str_buff);

  // Process and print out the array
  array_length = string_to_array(in_str_buff, input_array);
  if(array_length != 7) {
    Serial.println("Error: Please enter exactly 7 numbers.");
  } else {
    for (int i = 0; i < array_length; i++) {
      input->data.int8[i] = static_cast<int8>(input_arrau[i]);
   }
   // Run the model on this input and make sure it suceeds 
  if (kTflite0k != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  } else {
    TfliteTensor* output = interpreter->output(0);
    int8 prediction = output->data.int8[0];
    Serial.print("Model prediction: ");
    Serial.println(prediction);
  }
}
}

#include <Arduino.h>
#include <TensorFlowLite.h>
#include <sin.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_quantized_data.cc"


#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8

// put function declarations here:
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);


char received_char = (char)NULL;              
int chars_avail = 0;                    // input present on terminal
char out_str_buff[OUTPUT_BUFFER_SIZE];  // strings to print to terminal
char in_str_buff[INPUT_BUFFER_SIZE];    // stores input from terminal
int input_array[INT_ARRAY_SIZE];        // array of integers input by user

int in_buff_idx=0; // tracks current input location in input buffer
int array_length=0;
int array_sum=0;

// Assuming model_quantized_data.cc contains the quantized TFLite model
extern "C" { extern const unsigned char model_data[]; }
extern "C" { extern const unsigned int model_data_len; }

// namespace {
//   tflite::ErrorReporter* error_reporter;
//   const tflite::Model* model;
//   tflite::MicroInterpreter* interpreter;
//   TfLiteTensor* model_input;
//   TfLiteTensor* model_output;
  
//   // Define tensor arena size. Adjust according to the model's memory requirements.
//   constexpr int kTensorArenaSize = 2 * 1024;  
//   uint8_t tensor_arena[kTensorArenaSize];
// }

void setup() {
  // put your setup code here, to run once:
  delay(5000);
  // Arduino does not have a stdout, so printf does not work easily
  // So to print fixed messages (without variables), use 
  // Serial.println() (appends new-line)  or Serial.print() (no added new-line)
  Serial.println("Test Project waking up");
  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
}

void loop() {
  // Check if characters are available on the terminal input
  chars_avail = Serial.available();
  if (chars_avail > 0) {
    received_char = Serial.read(); // Get the typed character
    Serial.print(received_char);   // Echo to the terminal

    // Check if Enter key was pressed
    if (received_char == '\n' || received_char == '\r') {
      // Process the input line if Enter is pressed
      in_str_buff[in_buff_idx] = '\0'; // Null-terminate the string
      array_length = string_to_array(in_str_buff, input_array);

      // Verify exactly 7 numbers were entered
      if (array_length != 7) {
        Serial.println("Warning: Please enter exactly 7 numbers.");
      } else {
        // Measure printing time
        unsigned long t0 = micros();
        Serial.println("Processing...");
        unsigned long t1 = micros();

        // Prepare model input
        for (int i = 0; i < 7; i++) {
          // Assuming model_input is a tensor expecting int8 data
          model_input->data.int8[i] = input_array[i];
        }

        // Run model inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status == kTfLiteOk) {
          // Assuming model output is a single int8 value
          int8_t output_value = model_output->data.int8[0];
          Serial.print("Model prediction: ");
          Serial.println(static_cast<int>(output_value));
        } else {
          Serial.println("Error during model inference.");
        }
        unsigned long t2 = micros();

        // Print execution times
        Serial.print("Printing time = ");
        Serial.print(t1 - t0);
        Serial.println(" us.");
        Serial.print("Inference time = ");
        Serial.print(t2 - t1);
        Serial.println(" us.");
      }

      // Reset buffer and index for next input
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    } else if (in_buff_idx < INPUT_BUFFER_SIZE - 1) {
      // Add character to buffer if it's not Enter and buffer is not full
      in_str_buff[in_buff_idx++] = received_char;
    }
  }
}

void print_int_array(int *int_array, int array_len) {
  int curr_pos = 0; // track where in the output buffer we're writing

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff); // so the next write adds to the end
  for(int i=0;i<array_len;i++) {
    // sprintf returns number of char's written. use it to update current position
    curr_pos += sprintf(out_str_buff+curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff+curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}

int sum_array(int *int_array, int array_len) {
  int curr_sum = 0; // running sum of the array

  for(int i=0;i<array_len;i++) {
    curr_sum += int_array[i];
  }
  return curr_sum;
}
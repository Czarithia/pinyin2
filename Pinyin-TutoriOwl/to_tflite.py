import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# Ensure that STFT or other custom layers are defined or imported if needed
class STFT(Layer):
    # Define your STFT layer if necessary
    pass

# Define the path to the .h5 model file
h5_model_path = r'C:\Users\Prinz\Downloads\pinyin2\saved_models\96-71\conv2d_best_model.h5'
tflite_model_path = r'C:\Users\Prinz\Downloads\pinyin2\saved_models\96-71\conv2d.tflite'

# Load the model with the custom layer
custom_objects = {'STFT': STFT,
                    'Magnitude': Magnitude,
                    'ApplyFilterbank': ApplyFilterbank,
                    'MagnitudeToDecibel': MagnitudeToDecibel}
model = load_model(h5_model_path, custom_objects=custom_objects)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted and saved to", tflite_model_path)

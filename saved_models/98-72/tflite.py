import tensorflow as tf
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

#load h5 module
model = tf.keras.models.load_model('conv2d.h5', custom_objects={'STFT': STFT,
                        'Magnitude': Magnitude,
                        'ApplyFilterbank': ApplyFilterbank,
                        'MagnitudeToDecibel': MagnitudeToDecibel})
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)

#convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("conv2d.tflite", "wb") as f:
    f.write(tflite_model)
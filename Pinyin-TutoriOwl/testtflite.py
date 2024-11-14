import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import librosa
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/models/model.tflite")
interpreter.allocate_tensors()
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)


# Load and preprocess the audio file
def preprocess_audio(file_path, sample_rate=16000, duration=1.0):
    # Load audio with Librosa (ensure it matches the sample rate your model expects)
    wav, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    # Reshape or pad to ensure correct length
    wav = librosa.util.fix_length(wav, int(sample_rate * duration))
    return wav.reshape(-1, 1)  # Reshape to match model's input shape

# Processed input ready for the model
input_data = preprocess_audio(r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest1.wav")
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Add batch dimension

# Set the model input
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the model output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output_data)

predicted_class = np.argmax(output_data, axis=-1)
print("Predicted class:", predicted_class)
print("Prediction confidence:", output_data[0, predicted_class])

# Assuming labels is a list of class names
labels = ['a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f',
                'g', 'h', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'j', 'k',
                'l', 'm', 'n', 'o', 'ong', 'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan',
                'uang', 'ui', 'un', 'uo', 'uu', 'uuan', 'uue', 'uun', 'x', 'z', 'zh'] 
print("Predicted label:", labels[predicted_class[0]])

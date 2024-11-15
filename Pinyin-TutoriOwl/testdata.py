import tensorflow as tf
import numpy as np
import librosa

correct_predictions = 0
interpreter = tf.lite.Interpreter(model_path=r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/models/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f',
          'g', 'h', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'j', 'k',
          'l', 'm', 'n', 'o', 'ong', 'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan',
          'uang', 'ui', 'un', 'uo', 'uu', 'uuan', 'uue', 'uun', 'x', 'z', 'zh']

def preprocess_audio(file_path, sample_rate=16000, duration=1.0):
    wav, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    wav = librosa.util.fix_length(wav, int(sample_rate * duration))
    return wav.reshape(-1, 1)

audio_files = [
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest1.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest2.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest3.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest4.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/zh/zh_test.wav", "zh"),
]

total_files = len(audio_files)
for file_path, actual_label in audio_files:
    input_data = preprocess_audio(file_path)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=-1)
    predicted_label = labels[predicted_class[0]]
    
    print("File:", file_path)
    print("Model output:", output_data)
    print("Predicted class:", predicted_class)
    print("Prediction confidence:", output_data[0, predicted_class])
    print("Predicted label:", labels[predicted_class[0]])
    print("Actual label:", actual_label)
    print("********************************")
    if predicted_label == actual_label:
        correct_predictions += 1

print(f"Correct predictions: {correct_predictions} out of {total_files}")
accuracy = (correct_predictions / total_files) * 100
print(f"Accuracy: {accuracy:.2f}%")

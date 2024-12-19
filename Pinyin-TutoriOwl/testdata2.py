import tensorflow as tf
import numpy as np
import librosa

interpreter = tf.lite.Interpreter(model_path=r"C:\Users\Prinz\Downloads\pinyin2\Pinyin-TutoriOwl\models\model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng', 'er', 'f',
          'g', 'h', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iou', 'j', 'k',
          'l', 'm', 'n', 'o', 'ong', 'ou', 'p', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai', 'uan',
          'uang', 'ui', 'un', 'uo', 'uu', 'uuan', 'uue', 'uun', 'x', 'z', 'zh']

audio_files = [
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest1.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest2.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest3.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ang/angtest4.wav", "ang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/zh/zh_test.wav", "zh"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ai/ai.wav", "ai"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/ao/ao.wav", "ao"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/iong/iong.wav", "iong"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/j/j.wav", "j"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/uan/uan.wav", "uan"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/test_data/uang/uang.wav", "uang"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/clean/ai/ai_1_0.wav", "ai"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/clean/ai/ai_2_0.wav", "ai"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/clean/ai/ai_3_0.wav", "ai"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/clean/ai/ai_4_0.wav", "ai"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/clean/ai/ai_5_0.wav", "ai"),
    (r"C:/Users/Prinz/Downloads/pinyin2/Pinyin-TutoriOwl/clean/ai/ai_6_0.wav", "ai"),

]

def preprocess_audio(file_path, sample_rate=16000, duration=1.0):
    wav, _ = librosa.load(file_path, sr=sample_rate, duration=duration)
    wav = librosa.util.fix_length(wav, int(sample_rate * duration))
    return np.expand_dims(wav.reshape(-1, 1), axis=0).astype(np.float32)

def predict(file_path):
    input_data = preprocess_audio(file_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=-1)[0]
    return labels[predicted_class], output_data[0, predicted_class]

correct_predictions = 0
for file_path, actual_label in audio_files:
    predicted_label, confidence = predict(file_path)
    is_correct = predicted_label == actual_label
    correct_predictions += is_correct
    print(f"File: {file_path}\n"
          f"Predicted: {predicted_label} (Confidence: {confidence:.5f})\n"
          f"Actual: {actual_label}\n"
          f"Correct: {is_correct}\n{'-'*30}")

total_files = len(audio_files)
accuracy = (correct_predictions / total_files) * 100
print(f"Correct predictions: {correct_predictions} out of {total_files}")
print(f"Accuracy: {accuracy:.2f}%")

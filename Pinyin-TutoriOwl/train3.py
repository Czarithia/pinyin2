from kapre import STFTTflite, MagnitudeTflite
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from glob import glob
import os
import argparse
import warnings
from models import Conv1D, Conv2D, LSTM

def save_tflite_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(save_path, 'wb') as f:
        f.write(tflite_model)
        
def create_tflite_compatible_model(weights, input_shape, N_CLASSES=56):
    model_tflite = Sequential()
    model_tflite.add(STFTTflite(n_fft=512, win_length=400, hop_length=160,
                                window_name=None, pad_end=True,
                                input_data_format='channels_last',
                                output_data_format='channels_last',
                                input_shape=input_shape))
    model_tflite.add(MagnitudeTflite())
    model_tflite.add(LayerNormalization(axis=2, name='batch_norm'))
    model_tflite.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh', padding='same', name='td_conv_2d_tanh'))
    model_tflite.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1'))
    model_tflite.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv2d_relu_2'))
    model_tflite.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_3'))
    model_tflite.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2'))
    model_tflite.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_4'))
    model_tflite.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_5'))
    model_tflite.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3'))
    model_tflite.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_6'))
    model_tflite.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4'))
    model_tflite.add(layers.Conv2D(254, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_7'))
    model_tflite.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_5'))
    model_tflite.add(layers.Dropout(rate=0.25))
    model_tflite.add(layers.Flatten())
    model_tflite.add(layers.Dense(254, activation='relu', activity_regularizer=l2(0.001), name='dense1'))
    model_tflite.add(layers.Dense(128, activation='relu', activity_regularizer=l2(0.001), name='dense2'))
    model_tflite.add(layers.Dense(N_CLASSES, activation='softmax', name='softmax'))

    for i, layer in enumerate(model_tflite.layers):
        try:
            layer.set_weights(weights[i])
        except Exception as e:
            print(f"Layer {layer.name} - Error: {e}")

    model_tflite.summary()
    return model_tflite



def save_tflite_model(model, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(save_path, 'wb') as f:
        f.write(tflite_model)
        

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES':len(os.listdir(args.src_root)),
              'SR':sr,
              'DT':dt}
    models = {'conv1d':Conv1D(**params),
              'conv2d':Conv2D(**params),
              'lstm':  LSTM(**params)}
    assert model_type in models.keys(), '{} not an available model'.format(model_type)
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.1,
                                                                  random_state=0)

    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(len(set(label_val)), params['N_CLASSES']))

    tg = DataGenerator(wav_train, label_train, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    model = models[model_type]
    cp = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=True,
                         mode='auto', save_freq='epoch', verbose=1)
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(tg, validation_data=vg,
              epochs=30, verbose=1,
              callbacks=[csv_logger, cp])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite format')
    parser.add_argument('--model_type', type=str, default='conv2d',
                        help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--sample_rate', type=int, default=16000, 
                        help='Sampling rate of the audio')
    parser.add_argument('--delta_time', type=float, default=1.0, 
                        help='Duration of the audio sample')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--src_root', type=str, default='clean',
                        help='Path to the directory with label folders')

    args = parser.parse_args()
    train(args)
    model = Conv2D(N_CLASSES=56, SR=args.sample_rate, DT=args.delta_time)
    for layer in model.layers:
        print(f"Layer: {layer.name}, Shape: {[w.shape for w in layer.get_weights()]}")
    weights = model.get_weights()
    input_shape = (int(args.sample_rate * args.delta_time), 1)

    model_tflite = create_tflite_compatible_model(weights, input_shape, N_CLASSES=56)

save_tflite_model(model_tflite, r'C:\Users\Prinz\Downloads\pinyin2\Pinyin-TutoriOwl\models\model.tflite')
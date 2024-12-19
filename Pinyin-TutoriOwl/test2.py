from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm

def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT': STFT,
                        'Magnitude': Magnitude,
                        'ApplyFilterbank': ApplyFilterbank,
                        'MagnitudeToDecibel': MagnitudeToDecibel})

    wav_paths = glob(os.path.join(args.src_dir, '**/*.wav'), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths])

    classes = sorted([d for d in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, d))])
    le = LabelEncoder()
    le.fit(classes)

    for wav_fn in tqdm(wav_paths, desc="Classifying WAV files"):

        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr * args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i + step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)

        X_batch = np.array(batch, dtype=np.float32)

        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        predicted_index = np.argmax(y_mean)

        if predicted_index < len(classes):
            predicted_label = le.inverse_transform([predicted_index])[0]
            print(f'File: {os.path.basename(wav_fn)}, Predicted label: {predicted_label}')
        else:
            print(f'File: {os.path.basename(wav_fn)}, Predicted label: Unknown (index {predicted_index} out of bounds)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Prediction')
    parser.add_argument('--model_fn', type=str, default='models/conv2d.h5',
                        help='Path to the trained model file')
    parser.add_argument('--src_dir', type=str, default='test_data',
                        help='Directory containing subdirectories with WAV files')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='Time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate of clean audio')
    parser.add_argument('--threshold', type=int, default=200,
                        help='Threshold magnitude for np.int16 dtype')
    args = parser.parse_args()

    make_prediction(args)

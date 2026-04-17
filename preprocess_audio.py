import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def audio_to_melspectrogram(audio_path, output_path, shape=(224, 224)):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=shape[0])
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB_resized = cv2.resize(S_dB, shape)
    S_dB_norm = (S_dB_resized - S_dB_resized.min()) / (S_dB_resized.max() - S_dB_resized.min())
    S_dB_uint8 = (S_dB_norm * 255).astype(np.uint8)
    plt.imsave(output_path, S_dB_uint8, cmap='gray')

def preprocess_audio_dir(input_dir, output_dir, shape=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for file in files:
            if file.lower().endswith('.wav'):
                in_path = os.path.join(root, file)
                out_path = os.path.join(out_root, file + '.png')
                audio_to_melspectrogram(in_path, out_path, shape=shape)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess audio to Mel-spectrogram images')
    parser.add_argument('--input', required=True, help='Input audio directory')
    parser.add_argument('--output', required=True, help='Output image directory')
    parser.add_argument('--shape', type=int, nargs=2, default=[224,224], help='Target image shape')
    args = parser.parse_args()
    preprocess_audio_dir(args.input, args.output, tuple(args.shape))

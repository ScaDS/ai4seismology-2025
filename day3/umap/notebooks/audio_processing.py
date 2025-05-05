import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings

warnings.filterwarnings("ignore")

from config import SR, DURATION, N_MELS, N_FFT, HOP_LENGTH, N_MFCC

def extract_melspectrogram(
    audio_path,
    sr=SR,
    duration=DURATION,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
):
    """Extracts a mel-spectrogram from an audio file"""
    try:
        # Load audio with a fixed length
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)

        # If the audio is shorter than DURATION seconds, pad it with zeros
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)), "constant")

        # Extract the mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )

        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return mel_spectrogram_db
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def show_mel_augmentations(
        metadata, 
        data_path, 
        N_MELS=N_MELS,
        N_FFT=N_FFT,
        HOP_LENGTH=HOP_LENGTH):
    """Visualizes examples of mel-spectrograms for several classes"""
    classes = [0, 3, 8]  # select a few classes for the example

    plt.figure(figsize=(12, 6))
    for i, class_id in enumerate(classes):
        # select a random file from the chosen class
        sample = metadata[metadata["classID"] == class_id].sample(1).iloc[0]
        fold = sample["fold"]
        filename = sample["slice_file_name"]
        class_name = sample["class"]

        # path to the audio file
        audio_path = os.path.join(data_path, f"fold{fold}", filename)

        # load the audio
        y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)

        # if the audio is shorter than DURATION seconds, pad it with zeros
        if len(y) < sr * DURATION:
            y = np.pad(y, (0, sr * DURATION - len(y)), "constant")

        # extract the mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # display the waveform
        plt.subplot(3, 3, i * 3 + 1)
        plt.plot(y)
        plt.title(f"{class_name} - Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        # display the spectrogram
        plt.subplot(3, 3, i * 3 + 2)
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max
        )
        librosa.display.specshow(
            D, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="log"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{class_name} - Spectrogram")

        # display the mel-spectrogram
        plt.subplot(3, 3, i * 3 + 3)
        librosa.display.specshow(
            mel_spectrogram_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{class_name} - Mel-Spectrogram")

    plt.tight_layout()
    plt.show()

def extract_audio_features(audio_path, audio_dict_keys, sr=SR, duration=DURATION, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Extract only the requested audio features from an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        audio_dict_keys (set): Set of keys in audio_dict to determine which features to extract
        sr (int): Sample rate
        duration (float): Duration in seconds
        n_mfcc (int): Number of MFCC coefficients
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        
    Returns:
        dict: Dictionary containing only the requested extracted features
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Pad if necessary
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)), "constant")
    
    features = {}
    
    # Always include raw audio
    if "raw" in audio_dict_keys:
        features["raw"] = y
        
    # Calculate spectrogram if needed for any spectral features
    need_spectrogram = any(k in audio_dict_keys for k in ["rms"])
    
    S = None
    if need_spectrogram or "mean_mfccs" in audio_dict_keys:
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), 
            ref=np.max
        )
    
    # Extract only requested features
    if "rms" in audio_dict_keys:
        features["rms"] = librosa.feature.rms(S=S)[0]
        
    if "spec_bw" in audio_dict_keys:
        features["spec_bw"] = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
    if "poly_features" in audio_dict_keys:
        features["poly_features"] = librosa.feature.poly_features(y=y, sr=sr, order=0)[0]
        
    if "spec_centroid" in audio_dict_keys:
        features["spec_centroid"] = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
    if "spec_flatness" in audio_dict_keys:
        features["spec_flatness"] = librosa.feature.spectral_flatness(y=y)[0]
        
    if "spec_rolloff" in audio_dict_keys:
        features["spec_rolloff"] = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.5)[0]
        
    if "mean_mfccs" in audio_dict_keys:
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        )
        features["mean_mfccs"] = np.mean(mfccs.T, axis=0)
    
    return features
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

from audio_processing import extract_melspectrogram
from config import SR, DURATION, N_MELS, HOP_LENGTH



class AudioDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None):
        self.metadata = metadata
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fold = row["fold"]
        filename = row["slice_file_name"]
        class_id = row["classID"]

        # path to the audio file
        audio_path = os.path.join(self.root_dir, f"fold{fold}", filename)

        # extract the mel-spectrogram
        mel_spectrogram = extract_melspectrogram(audio_path)

        # if the spectrogram could not be extracted, return a zero matrix
        if mel_spectrogram is None:
            mel_spectrogram = np.zeros((N_MELS, int(SR * DURATION / HOP_LENGTH) + 1))

        # normalize
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / (
            mel_spectrogram.std() + 1e-8
        )

        # apply transformations if any
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        # convert to tensor
        mel_spectrogram = torch.FloatTensor(mel_spectrogram).unsqueeze(
            0
        )  # add channel

        return mel_spectrogram, class_id


class SpectrogramAugmentation:
    def __init__(self, time_mask_param=30, freq_mask_param=20, p=0.5):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.p = p

    def __call__(self, spec):
        """Apply time and frequency masking augmentation."""
        augmented_spec = np.array(spec, copy=True)  # Ensure it's a NumPy array copy

        # Time Masking
        if np.random.rand() < self.p:
            t = np.random.randint(1, self.time_mask_param)  # Avoid zero-length mask
            t0 = np.random.randint(
                0, max(1, augmented_spec.shape[1] - t)
            )  # Ensure valid range
            augmented_spec[:, t0 : t0 + t] = 0

        # Frequency Masking
        if np.random.rand() < self.p:
            f = np.random.randint(1, self.freq_mask_param)  # Avoid zero-length mask
            f0 = np.random.randint(
                0, max(1, augmented_spec.shape[0] - f)
            )  # Ensure valid range
            augmented_spec[f0 : f0 + f, :] = 0

        return augmented_spec


import os
import uuid
import librosa
import numpy as np
import librosa.display
from scipy.signal import welch
import IPython.display as ipd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Audio


class AudioVis:
    """Class for audio visualizations."""

    def __init__(self, data_dir: str = './data'):
        """Initialize the class with a directory to save data."""
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def play_audio(self, samples: np.array, sr: int = 22000) -> None:
        """Plays the provided audio samples."""
        audio_path = os.path.join(self.data_dir, f"{uuid.uuid4()}.wav")
        librosa.output.write_wav(audio_path, samples, sr)
        display(Audio(filename=audio_path))

    def _save_and_display_plot(self, title: str) -> None:
        """Save the current plot as an image and display it."""
        img_path = os.path.join(self.data_dir, f"{uuid.uuid4()}.png")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        display(Image(filename=img_path))

    def wav_plot(self, signal: np.array, sr: int = 22000, title: str = "Audio Signal") -> None:
        """Plots the waveform of the audio signal."""
        plt.figure(figsize=(15, 4))
        plt.plot(signal)
        plt.title(title)
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        self._save_and_display_plot(title)

    def plot_spectrogram(self, signal: np.array, sr: int = 22000, title: str = "Spectrogram") -> None:
        """Displays a spectrogram of the audio signal."""
        X = librosa.stft(signal)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.title(title)
        self._save_and_display_plot(title)

import os
import uuid
import librosa
import numpy as np
import soundfile as sf
import librosa.display
from scipy.signal import welch
import IPython.display as ipd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Audio, HTML
import io
import contextlib
import base64

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
        sf.write(audio_path, samples, sr)
        audio_link = f'<audio controls style="width: 100%"><source src="{audio_path}" type="audio/wav"></audio>'
        display(HTML(audio_link))

    def _save_and_display_plot(self, title: str) -> None:
        """Save the current plot as an image and display it using HTML."""
        img_path = os.path.join(self.data_dir, f"{uuid.uuid4()}.png")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        img_link = f'<img src="{img_path}" alt="{title}" width="100%"/>'
        display(HTML(img_link))

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

    def plot_log_spectrogram(self, signal: np.array, sr: int = 22000, title: str = "Log Spectrogram") -> None:
        """Displays a logarithmic spectrogram of the audio signal."""
        X = librosa.stft(signal)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar()
        plt.title(title)
        self._save_and_display_plot(title)

    def plot_mfcc(self, signal: np.array, sr: int = 22000, title: str = "MFCC", n_mfcc: int = 13) -> None:
        """Visualizes the MFCC of the audio signal."""
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(mfccs, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title(title)
        self._save_and_display_plot(title)

    def plot_psd(self, signal: np.array, sr: int = 22000, title: str = "Power Spectral Density") -> None:
        """Plots the power spectral density of the audio signal."""
        freqs, psd = welch(signal, fs=sr)
        plt.figure(figsize=(15, 5))
        plt.semilogy(freqs, psd)
        plt.title(title)
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        self._save_and_display_plot(title)

    def display_collapsible(self, debug_info, title="Debugging Information"):
        """Displays a collapsible/expandable section in Jupyter Notebook."""
        # Create the collapsible section with a button
        collapsible_html = f"""
        <details>
            <summary>{title}</summary>
            {'<br>'.join(debug_info)}
        </details>
        """
        display(HTML(collapsible_html))

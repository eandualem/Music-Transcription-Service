import librosa
import numpy as np

class AudioScorer:
    def __init__(self, original_audio):
        self.original_audio = original_audio

    def amplitude_matching_score(self, user_audio, original_audio):
        """Compute a score based on amplitude matching."""
        difference = user_audio - original_audio
        return 1 / (1 + np.mean(np.abs(difference)))

    def spectral_matching_score(self, user_audio, original_audio):
        """Compute a score based on spectral matching."""
        user_spectrogram = np.abs(librosa.stft(user_audio))
        original_spectrogram = np.abs(librosa.stft(original_audio))
        difference = user_spectrogram - original_spectrogram
        return 1 / (1 + np.mean(np.abs(difference)))

    def mfcc_matching_score(self, user_audio, original_audio):
        """Compute a score based on MFCC matching."""
        user_mfccs = librosa.feature.mfcc(user_audio)
        original_mfccs = librosa.feature.mfcc(original_audio)
        difference = user_mfccs - original_mfccs
        return 1 / (1 + np.mean(np.abs(difference)))

    def combined_score(self, amplitude_score, spectral_score, mfcc_score):
        """Compute a combined score based on weights."""
        weights = {
            'amplitude': 0.3,
            'spectral': 0.3,
            'mfcc': 0.4
        }
        combined = (weights['amplitude'] * amplitude_score +
                    weights['spectral'] * spectral_score +
                    weights['mfcc'] * mfcc_score)
        return combined

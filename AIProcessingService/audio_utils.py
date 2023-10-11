import numpy as np
import librosa

class AudioUtils:
    @staticmethod
    def align_audio(user_audio, original_audio):
        """Align user's audio chunk with the original audio using DTW."""
        user_mfccs = librosa.feature.mfcc(user_audio)
        original_mfccs = librosa.feature.mfcc(original_audio)

        # Compute the DTW alignment between the two MFCC sequences
        _, path = librosa.sequence.dtw(user_mfccs, original_mfccs)

        # Extract the aligned audio chunk based on DTW path
        aligned_audio_chunk = user_audio[path[0]]

        return aligned_audio_chunk

    @staticmethod
    def is_noisy(audio):
        """Determine if the audio chunk is too noisy."""
        signal_power = np.mean(audio**2)
        noise_power = np.mean((audio - np.mean(audio))**2)

        if noise_power == 0:  # Avoid division by zero
            return False

        snr = 10 * np.log10(signal_power / noise_power)

        # Threshold below which audio is considered noisy
        SNR_THRESHOLD = 10  # This value can be adjusted based on testing
        return snr < SNR_THRESHOLD

import librosa
import numpy as np
from typing import Optional, List
from inspect import signature
from scipy.signal import butter, filtfilt
import scipy.signal


class AudioPreprocessor:
    """Handles audio preprocessing tasks."""

    EPSILON = 1e-10  # Global constant to avoid division by zero
    DEFAULT_TRIM_DB = 20

    @staticmethod
    def normalize(signal: np.array) -> np.array:
        """Normalizes a given audio segment."""
        feats_mean = np.mean(signal)
        feats_std = np.std(signal)
        return (signal - feats_mean) / (feats_std + AudioPreprocessor.EPSILON)

    @staticmethod
    def trim_audio(signal: np.array, trim_db: float = DEFAULT_TRIM_DB) -> np.array:
        """Trims leading and trailing silence from an audio signal."""
        signal, _ = librosa.effects.trim(signal, top_db=trim_db)
        return signal

    @staticmethod
    def split_audio(signal: np.array, clean_db: float = DEFAULT_TRIM_DB) -> np.array:
        """Splits audio into non-silent chunks."""
        intervals = librosa.effects.split(signal, top_db=clean_db)
        cleaned_signal = [signal[start:end] for start, end in intervals]
        return np.concatenate(cleaned_signal, axis=0)

    @staticmethod
    def butter_lowpass(cutoff_freq: float, sample_rate: int, order: int = 5) -> np.array:
        """Designs a low-pass Butterworth filter."""
        nyq = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    @staticmethod
    def apply_lowpass(signal: np.array, cutoff_freq: float, sample_rate: int) -> np.array:
        """Applies a low-pass filter to the given signal."""
        b, a = AudioPreprocessor.butter_lowpass(cutoff_freq, sample_rate)
        y = filtfilt(b, a, signal)
        return y

    @staticmethod
    def dynamic_range_compression(segment: np.array, threshold: float = 0.02) -> np.array:
        """Applies dynamic range compression to an audio segment."""
        # Set values below threshold to 0
        compressed_segment = np.where(np.abs(segment) < threshold, 0, segment)
        return compressed_segment

    @staticmethod
    def bandpass_filter(signal: np.array, sr: int, lowcut: float = 20.0, highcut: float = 20000.0) -> np.array:
        """Applies a bandpass filter to keep frequencies within the human hearing range."""
        nyq = 0.5 * sr  # Nyquist frequency, which is half of the sample rate
        low = max(0.01, lowcut / nyq)  # Ensure low is within range
        high = min(0.99, highcut / nyq)  # Ensure high is within range

        if high <= low:
            raise ValueError("Highcut frequency must be greater than lowcut frequency")

        b, a = scipy.signal.butter(6, [low, high], btype="band")
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        return filtered_signal

    @staticmethod
    def spectral_gate(signal: np.array, threshold: float = 0.1, alpha: float = 2.0) -> np.array:
        """
        Suppresses frequency components of the signal below a dynamic threshold.

        Parameters:
            signal (np.array): The audio signal to process.
            threshold (float): A scaling factor for the dynamic threshold.
            alpha (float): Exponent for magnitude squaring.

        Returns:
            np.array: The processed audio signal.
        """
        # Compute the STFT of the signal
        stft_signal = librosa.stft(signal)
        magnitude, phase = librosa.magphase(stft_signal)

        # Square the magnitudes to emphasize stronger components
        magnitude_squared = magnitude**alpha

        # Compute a dynamic threshold based on the mean and standard deviation of the squared magnitudes
        dynamic_threshold = threshold * (np.mean(magnitude_squared) + np.std(magnitude_squared))

        # Create a soft mask based on the dynamic threshold
        mask = np.tanh(magnitude_squared / dynamic_threshold)

        # Apply the mask to the original STFT
        masked_stft = stft_signal * mask

        # Inverse STFT to obtain the time-domain signal
        processed_signal = librosa.istft(masked_stft)

        return processed_signal

    @staticmethod
    def spectral_masking(audio_chunk: np.array, reference_audio: np.array) -> np.array:
        """Applies spectral masking based on reference audio."""

        print(f"audio_chunk: {audio_chunk.shape}")
        print(f"reference_audio: {reference_audio.shape}")
        user_stft = librosa.stft(audio_chunk)
        ref_stft = librosa.stft(reference_audio)

        user_magnitude = np.abs(user_stft)
        ref_magnitude = np.abs(ref_stft)

        # Normalize magnitudes
        user_magnitude /= np.max(user_magnitude) + 1e-10
        ref_magnitude /= np.max(ref_magnitude) + 1e-10

        # Wiener mask
        wiener_mask = user_magnitude**2 / (user_magnitude**2 + ref_magnitude**2 + 1e-10)

        masked_stft = user_stft * wiener_mask
        masked_audio = librosa.istft(masked_stft)

        return masked_audio

    @staticmethod
    def voice_activity_detection(
        signal: np.array, sampling_rate: int, hop_length: int = 512, top_db: float = 20
    ) -> np.array:
        """Retains segments of the signal with vocal activity."""
        S = np.abs(librosa.stft(signal, hop_length=hop_length))
        intervals = librosa.effects.split(S, top_db=top_db)
        voiced_signal = [signal[start:end] for start, end in intervals]
        return np.concatenate(voiced_signal, axis=0)

    @staticmethod
    def adaptive_noise_reduction(audio_chunk: np.array, reference_audio: np.array, sr: int = 22050) -> np.array:
        """
        Reduces noise in the audio chunk using the reference audio to model the noise.
        Assumes that the length of the audio chunk and reference audio are the same.
        """
        # Compute the Short-Time Fourier Transform (STFT) of both audio signals
        audio_stft = librosa.stft(audio_chunk)
        reference_stft = librosa.stft(reference_audio)

        # Estimate the noise profile from the reference audio
        noise_profile = np.mean(np.abs(reference_stft), axis=1, keepdims=True)

        # Subtract the noise profile from the audio chunk's magnitude spectrum
        magnitude, phase = librosa.magphase(audio_stft)
        magnitude -= noise_profile
        magnitude = np.maximum(magnitude, 0)  # Ensure that magnitude values are non-negative

        # Reconstruct the denoised audio using the modified magnitude and original phase
        denoised_stft = magnitude * phase
        denoised_audio = librosa.istft(denoised_stft)

        return denoised_audio

    @staticmethod
    def wiener_filter(audio_chunk: np.array, background_track: np.array) -> np.array:
        """
        Applies a Wiener filter to separate vocals from background.
        :param audio_chunk: The mixed audio data.
        :param background_track: The background audio to estimate noise power.
        :return: The separated vocal audio data.
        """
        mixed_stft = librosa.stft(audio_chunk)
        background_stft = librosa.stft(background_track)

        mixed_magnitude, mixed_phase = librosa.magphase(mixed_stft)
        background_magnitude = np.abs(background_stft)

        # Estimate power spectral densities
        signal_psd = mixed_magnitude**2
        noise_psd = background_magnitude**2

        # Wiener filter
        wiener_filter = signal_psd / (signal_psd + noise_psd + AudioPreprocessor.EPSILON)

        # Apply the filter and reconstruct audio
        vocal_stft = mixed_stft * wiener_filter
        vocal_audio = librosa.istft(vocal_stft)

        return vocal_audio

    @staticmethod
    def preprocess_audio(audio: np.array, pipeline: List[str], **kwargs) -> np.array:
        """Processes the audio through the specified preprocessing steps."""
        processing_map = {
            "normalize": AudioPreprocessor.normalize,
            "trim_silences": AudioPreprocessor.trim_audio,
            "split_audio": AudioPreprocessor.split_audio,
            "apply_lowpass": AudioPreprocessor.apply_lowpass,
            "dynamic_range_compression": AudioPreprocessor.dynamic_range_compression,
            "bandpass_filter": AudioPreprocessor.bandpass_filter,
            "spectral_gate": AudioPreprocessor.spectral_gate,
            "spectral_masking": AudioPreprocessor.spectral_masking,
            "voice_activity_detection": AudioPreprocessor.voice_activity_detection,
            "adaptive_noise_reduction": AudioPreprocessor.adaptive_noise_reduction,
            "wiener_filter": AudioPreprocessor.wiener_filter,
        }
        for step in pipeline:
            if step in processing_map:
                func = processing_map[step]

                # Filter kwargs based on the function's signature
                sig = signature(func)
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

                audio = func(audio, **filtered_kwargs)
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")
        return audio

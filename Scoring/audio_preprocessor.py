import librosa
import numpy as np
from typing import Optional, List
from inspect import signature


class AudioPreprocessor:
    """Handles audio preprocessing tasks."""

    DEFAULT_TRIM_DB = 20
    EPSILON = 1e-14  # Small value to avoid division by zero

    @staticmethod
    def _normalize_segment(segment: np.array) -> np.array:
        """Normalizes a given audio segment."""
        feats_mean = np.mean(segment)
        feats_std = np.std(segment)
        return (segment - feats_mean) / (feats_std + AudioPreprocessor.EPSILON)

    @staticmethod
    def normalize_audio(signal: np.array, segment_length: Optional[int] = None) -> np.array:
        """Normalizes the audio signal. If segment_length is provided, perform segment-wise normalization."""
        if segment_length:
            for i in range(0, len(signal), segment_length):
                signal[i : i + segment_length] = AudioPreprocessor._normalize_segment(signal[i : i + segment_length])
        else:
            signal = AudioPreprocessor._normalize_segment(signal)
        return signal

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
    def spectral_gate(signal: np.array, threshold: float = 0.1) -> np.array:
        """Suppresses frequency components of the signal below the threshold."""
        stft_signal = librosa.stft(signal)
        stft_signal[np.abs(stft_signal) < threshold] = 0
        return librosa.istft(stft_signal)

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
    def source_separation(audio_chunk: np.array, sr: int = 22050) -> np.array:
        """Separates the harmonic component using Harmonic/Percussive source separation."""
        # Separate harmonic and percussive components
        harmonic, _ = librosa.effects.hpss(audio_chunk)
        return harmonic

    @staticmethod
    def spectral_masking(audio_chunk: np.array, reference_audio: np.array) -> np.array:
        """Applies spectral masking based on reference audio."""
        if len(reference_audio) > len(audio_chunk):
            reference_audio = reference_audio[: len(audio_chunk)]
        user_stft = librosa.stft(audio_chunk)
        ref_stft = librosa.stft(reference_audio)
        user_magnitude = np.abs(user_stft)
        ref_magnitude = np.abs(ref_stft)
        mask = user_magnitude / (user_magnitude + ref_magnitude + 1e-10)
        masked_stft = user_stft * mask
        masked_audio = librosa.istft(masked_stft)
        return masked_audio

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
    def preprocess_audio(audio: np.array, pipeline: List[str], **kwargs) -> np.array:
        """Processes the audio through the specified preprocessing steps."""
        processing_map = {
            "normalize": AudioPreprocessor.normalize_audio,
            "trim_silences": AudioPreprocessor.trim_audio,
            "spectral_gate": AudioPreprocessor.spectral_gate,
            "adaptive_noise_reduction": AudioPreprocessor.adaptive_noise_reduction,
            "voice_activity_detection": AudioPreprocessor.voice_activity_detection,
            "source_separation": AudioPreprocessor.source_separation,
            "spectral_masking": AudioPreprocessor.spectral_masking,
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

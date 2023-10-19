import librosa
import logging
import numpy as np
from typing import Callable, Dict
from dtw_helper import DTWHelper
from Levenshtein import distance as levenshtein_distance
from audio_vis import AudioVis


class AudioScorer:
    """Computes various audio scores."""

    def __init__(self, transcriber: Callable, dtw_method: str = "fastdtw"):
        self.transcriber = transcriber
        self.dtw_helper = DTWHelper(method=dtw_method)
        self.av = AudioVis()

    @staticmethod
    def _levenshtein_similarity(text1: str, text2: str) -> float:
        """Compute Levenshtein similarity between two strings."""
        distance = levenshtein_distance(text1.lower().strip(), text2.lower().strip())
        return 1 - (distance / max(len(text1), len(text2)))

    def linguistic_accuracy_score(self, user_audio: np.ndarray, sr: int, actual_lyrics: str) -> float:
        """Linguistic accuracy based on transcribed text."""
        try:
            user_transcription = self.transcriber.transcribe(user_audio, sr)
            self.av.play_audio(user_audio, sr)
            print(f"transcription: {user_transcription}")
            print(f"actual_lyrics: {actual_lyrics}")
            return self._levenshtein_similarity(user_transcription, actual_lyrics)
        except Exception as e:
            logging.error(f"Linguistic accuracy computation failed: {e}")
            return 0.0

    def compute_dtw_score(self, user_audio_features: np.ndarray, reference_audio_features: np.ndarray) -> float:
        """DTW score between user and reference audio features."""
        return self.dtw_helper.compute_similarity(user_audio_features, reference_audio_features)

    def amplitude_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray, sr: int) -> float:
        """Amplitude matching score."""
        new_sample_rate = sr // 8

        # We might not need to downsample by this much if fastdtw works
        user_audio_downsampled = librosa.resample(user_audio, orig_sr=sr, target_sr=new_sample_rate)
        reference_audio_downsampled = librosa.resample(reference_audio, orig_sr=sr, target_sr=new_sample_rate)

        return self.compute_dtw_score(user_audio_downsampled.flatten(), reference_audio_downsampled.flatten())

    def pitch_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """Pitch matching score."""
        user_pitch, _, _ = librosa.pyin(user_audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        reference_pitch, _, _ = librosa.pyin(
            reference_audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        # Remove NaN values:
        user_pitch = user_pitch[~np.isnan(user_pitch)]
        reference_pitch = reference_pitch[~np.isnan(reference_pitch)]

        # return self.compute_dtw_score(user_pitch, reference_pitch) # Some bug here
        return self.dtw_helper.compute_similarity_dtaidistance(user_pitch, reference_pitch)

    def rhythm_score(self, user_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """Rhythm score."""
        user_onset_env = librosa.onset.onset_strength(y=user_audio)
        reference_onset_env = librosa.onset.onset_strength(y=reference_audio)
        return self.compute_dtw_score(user_onset_env, reference_onset_env)

    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        reference_audio: np.ndarray,
        actual_lyrics: str,
        sr: int,
    ) -> Dict[str, float]:
        """Compute scores for an audio chunk."""
        scoring_functions = {
            "linguistic_score": lambda x, y: self.linguistic_accuracy_score(x, sr, actual_lyrics),
            "amplitude_score": lambda x, y: self.amplitude_matching_score(x, y, sr),
            "pitch_score": lambda x, y: self.pitch_matching_score(x, y),
            "rhythm_score": lambda x, y: self.rhythm_score(x, y),
        }

        scores = {score_name: func(audio_chunk, reference_audio) for score_name, func in scoring_functions.items()}
        return scores

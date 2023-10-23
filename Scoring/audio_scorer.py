import logging
import librosa
import numpy as np
from Scoring.dtw_helper import DTWHelper
from typing import Callable, Dict
from Levenshtein import distance as levenshtein_distance
import logging


class AudioScorer:
    """Computes various audio scores."""

    def __init__(self, transcriber: Callable, dtw_method: str = "fastdtw"):
        self.transcriber = transcriber
        self.dtw_helper = DTWHelper(method=dtw_method)

        self.scoring_functions = {
            "linguistic_accuracy_score": self.linguistic_accuracy_score,
            "linguistic_similarity_score": self.linguistic_similarity_with_original,
            "amplitude_score": self.amplitude_matching_score,
            "pitch_score": self.pitch_matching_score,
            "rhythm_score": self.rhythm_score,
        }

    @staticmethod
    def _levenshtein_similarity(text1: str, text2: str) -> float:
        """Compute Levenshtein similarity between two strings."""
        distance = levenshtein_distance(text1.lower().strip(), text2.lower().strip())
        return 1 - (distance / max(len(text1), len(text2)))

    def linguistic_accuracy_score(self, user_audio: np.ndarray, **kwargs) -> float:
        """Linguistic accuracy based on transcribed text."""
        sr = kwargs.get("sr")
        actual_lyrics = kwargs.get("actual_lyrics")
        from_file = kwargs.get("from_file", False)

        try:
            user_transcription = self.transcriber.transcribe(user_audio, sr, from_file=from_file)
            return self._levenshtein_similarity(user_transcription, actual_lyrics)
        except Exception as e:
            logging.error(f"Linguistic accuracy computation failed: {e}")
            return 0.0

    def linguistic_similarity_with_original(
        self, user_audio: np.ndarray, reference_audio: np.ndarray, **kwargs
    ) -> float:
        """Compute linguistic similarity with the original singer's transcription."""
        sr = kwargs.get("sr")
        from_file = kwargs.get("from_file", False)

        user_transcription = self.transcriber.transcribe(user_audio, sr, from_file=from_file)
        original_transcription = self.transcriber.transcribe(reference_audio, sr, from_file=from_file)
        return self._levenshtein_similarity(user_transcription, original_transcription)

    def compute_dtw_score(self, user_audio_features: np.ndarray, reference_audio_features: np.ndarray) -> float:
        """DTW score between user and reference audio features."""
        return self.dtw_helper.compute_similarity(user_audio_features, reference_audio_features)

    def amplitude_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray, **kwargs) -> float:
        """Amplitude matching score."""
        sr = kwargs.get("sr")
        new_sample_rate = sr // 16
        user_audio_downsampled = librosa.resample(user_audio, orig_sr=sr, target_sr=new_sample_rate)
        reference_audio_downsampled = librosa.resample(reference_audio, orig_sr=sr, target_sr=new_sample_rate)
        return self.compute_dtw_score(user_audio_downsampled.flatten(), reference_audio_downsampled.flatten())

    def pitch_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray, **kwargs) -> float:
        """Pitch matching score."""
        user_pitch, _, _ = librosa.pyin(user_audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        reference_pitch, _, _ = librosa.pyin(
            reference_audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        user_pitch = user_pitch[~np.isnan(user_pitch)]
        reference_pitch = reference_pitch[~np.isnan(reference_pitch)]
        return self.dtw_helper.compute_similarity_dtaidistance(user_pitch, reference_pitch)

    def rhythm_score(self, user_audio: np.ndarray, reference_audio: np.ndarray, **kwargs) -> float:
        """Rhythm score."""
        user_onset_env = librosa.onset.onset_strength(y=user_audio)
        reference_onset_env = librosa.onset.onset_strength(y=reference_audio)
        return self.compute_dtw_score(user_onset_env, reference_onset_env)

    def process_audio_chunk(
        self,
        processed_audio_chunk_data: Dict[str, np.ndarray],
        processed_original_data: Dict[str, np.ndarray],
        actual_lyrics: str,
        sr: int,
        from_file: bool = False,
    ) -> Dict[str, float]:
        """Compute scores for an audio chunk."""
        scores = {}
        for score_name, scoring_function in self.scoring_functions.items():
            kwargs = {
                "sr": sr,
                "actual_lyrics": actual_lyrics,
                "reference_audio": processed_original_data[score_name],
                "from_file": from_file,
            }

            user_audio = processed_audio_chunk_data[score_name]
            scores[score_name] = scoring_function(user_audio, **kwargs)
            logging.info(f"\n\n{score_name} {scores}")
        return scores

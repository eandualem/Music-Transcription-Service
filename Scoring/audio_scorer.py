import os
import logging
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from fuzzywuzzy import fuzz
from typing import Callable, Dict
from tempfile import NamedTemporaryFile
from Scoring.dtw_helper import DTWHelper
from Levenshtein import distance as levenshtein_distance
from concurrent.futures import ThreadPoolExecutor, as_completed
import io


class AudioScorer:
    """Computes various audio scores."""

    def __init__(self, transcriber: Callable, config, dtw_method: str = "fastdtw"):
        self.transcriber = transcriber
        self.dtw_helper = DTWHelper(method=dtw_method)
        self.scoring_functions = {
            "linguistic_accuracy_score": self.linguistic_accuracy_score,
            "linguistic_similarity_score": self.linguistic_similarity_with_original,
            "amplitude_score": self.amplitude_matching_score,
            "pitch_score": self.pitch_matching_score,
            "rhythm_score": self.rhythm_score,
        }

    def _extract_kwargs(self, **kwargs) -> tuple:
        """Extract common keyword arguments."""
        sr = kwargs.get("sr")
        return sr

    @staticmethod
    def save_audio_to_file(audio_data: np.ndarray, sample_rate: int) -> str:
        # Create and manage a temporary file to store the audio
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_file_path = temp_file.name

        return temp_file_path

    @staticmethod
    def save_audio_to_memory(audio_data: np.ndarray, sample_rate: int) -> io.BytesIO:
        """
        Save the audio data to an in-memory file.
        """
        # Create a BytesIO object
        in_memory_file = io.BytesIO()

        # Write the audio data to the in-memory file
        sf.write(in_memory_file, audio_data, sample_rate, format="WAV")

        # Seek to the beginning of the in-memory file so it's ready for reading
        in_memory_file.seek(0)

        return in_memory_file

    @staticmethod
    def _levenshtein_similarity(text1: str, text2: str) -> float:
        """Compute Levenshtein similarity between two strings."""
        distance = levenshtein_distance(text1.lower().strip(), text2.lower().strip())
        return 1 - (distance / max(len(text1), len(text2)))

    @staticmethod
    def _lenient_similarity(text1: str, text2: str) -> float:
        """Compute lenient similarity between two strings using FuzzyWuzzy's partial_ratio."""
        return fuzz.partial_ratio(text1.lower().strip(), text2.lower().strip()) / 100.0

    def linguistic_accuracy_score(self, user_transcription: str, actual_lyrics: str) -> float:
        """Linguistic accuracy based on transcribed text."""
        logging.info(f"Respooons user_transcription: {user_transcription}")

        try:
            return self._lenient_similarity(user_transcription, actual_lyrics)
        except Exception as e:
            logging.error(f"Linguistic accuracy computation failed: {e}")
            return 0.0

    def linguistic_similarity_with_original(
        self,
        user_transcription: str,
        original_transcription: str,
    ) -> float:
        """Compute linguistic similarity with the original singer's transcription."""
        try:
            return self._lenient_similarity(user_transcription, original_transcription)
        except Exception as e:
            logging.error(f"Linguistic accuracy computation failed: {e}")
            return 0.0

    @staticmethod
    def _resample_audio(audio: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
        """Resample audio to the target sample rate."""
        return librosa.resample(audio, orig_sr=original_sample_rate, target_sr=target_sample_rate)

    def compute_dtw_score(
        self, user_audio_features: np.ndarray, reference_audio_features: np.ndarray, tolerance: float = None
    ) -> float:
        """DTW score between user and reference audio features."""
        return self.dtw_helper.compute_similarity(user_audio_features, reference_audio_features, tolerance)

    def amplitude_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray, **kwargs) -> float:
        """Amplitude matching score."""
        sr = self._extract_kwargs(**kwargs)

        scale = 16
        tolerance = 0.00005

        user_audio_downsampled = self._resample_audio(user_audio, sr, sr // scale)
        reference_audio_downsampled = self._resample_audio(reference_audio, sr, sr // scale)
        return self.compute_dtw_score(
            user_audio_downsampled.flatten(), reference_audio_downsampled.flatten(), tolerance
        )

    def pitch_matching_score(self, user_audio: np.ndarray, reference_audio: np.ndarray, **kwargs) -> float:
        """Pitch matching score."""
        sr = self._extract_kwargs(**kwargs)

        scale = 2
        tolerance = 0.5

        user_audio_resampled = self._resample_audio(user_audio, sr, sr // scale)
        reference_audio_resampled = self._resample_audio(reference_audio, sr, sr // scale)

        user_pitch, _, _ = librosa.pyin(
            user_audio_resampled, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        reference_pitch, _, _ = librosa.pyin(
            reference_audio_resampled, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )

        # Carry forward last observation to replace NaN values
        user_pitch = pd.Series(user_pitch).ffill().bfill().values
        reference_pitch = pd.Series(reference_pitch).ffill().bfill().values

        # Handling no pitch detected scenarios
        if np.all(np.isnan(user_pitch)) and np.all(np.isnan(reference_pitch)):
            # No pitch detected in both audios, return neutral score
            return 0.5
        elif np.all(np.isnan(user_pitch)):
            # No pitch detected only in user audio, return low score
            return 0.2  # You may want to adjust this value based on your scoring system

        return self.compute_dtw_score(user_pitch, reference_pitch, tolerance)

    def rhythm_score(self, user_audio: np.ndarray, reference_audio: np.ndarray, **kwargs) -> float:
        """Rhythm score."""
        sr = self._extract_kwargs(**kwargs)

        scale = 4
        tolerance = 0.05

        user_audio_resampled = self._resample_audio(user_audio, sr, sr // scale)
        reference_audio_resampled = self._resample_audio(reference_audio, sr, sr // scale)
        user_onset_env = librosa.onset.onset_strength(y=user_audio_resampled)
        reference_onset_env = librosa.onset.onset_strength(y=reference_audio_resampled)
        return self.compute_dtw_score(user_onset_env, reference_onset_env, tolerance)

    def parallel_transcription(self, user_audio, original_audio, sr):
        """Conducts parallel transcription for user and original audio."""
        user_audio_file = self.save_audio_to_file(user_audio, sr)
        original_audio_file = self.save_audio_to_file(original_audio, sr)
        with ThreadPoolExecutor() as executor:
            future_user_transcription = executor.submit(self.transcriber.transcribe, user_audio_file)
            future_original_transcription = executor.submit(self.transcriber.transcribe, original_audio_file)
        os.remove(user_audio_file)
        os.remove(original_audio_file)

        return future_user_transcription.result(), future_original_transcription.result()

    def score_function_wrapper(self, score_name, scoring_function, user_audio, **kwargs):
        """Wrapper function to execute scoring functions and handle exceptions."""
        try:
            score_value = scoring_function(user_audio, **kwargs)
            return score_name, score_value
        except Exception as e:
            logging.error(f"Error computing {score_name}: {e}")
            return score_name, 0.0

    def process_audio_chunk(
        self,
        processed_audio_chunk_data: Dict[str, np.ndarray],
        processed_original_data: Dict[str, np.ndarray],
        actual_lyrics: str,
        sr: int,
    ) -> Dict[str, float]:
        """Compute scores for an audio chunk in parallel."""
        scores = {}

        with ThreadPoolExecutor() as executor:
            # Dispatch the three scoring functions that do not require transcription.
            future_to_score = {
                executor.submit(
                    self.score_function_wrapper,
                    score_name,
                    scoring_function,
                    processed_audio_chunk_data[score_name],
                    sr=sr,
                    actual_lyrics=actual_lyrics,
                    reference_audio=processed_original_data[score_name],
                ): score_name
                for score_name, scoring_function in self.scoring_functions.items()
                if score_name in ["amplitude_score", "pitch_score", "rhythm_score"]
            }

            # Concurrently initiate the transcription task.
            future_transcription = executor.submit(
                self.parallel_transcription,
                processed_audio_chunk_data["linguistic_accuracy_score"],
                processed_original_data["linguistic_similarity_score"],
                sr,
            )

            # As transcription completes, dispatch the linguistic scoring functions.
            user_transcription, original_transcription = future_transcription.result()

            # Directly calling the linguistic scoring functions
            scores["linguistic_accuracy_score"] = self.linguistic_accuracy_score(user_transcription, actual_lyrics)
            scores["linguistic_similarity_score"] = self.linguistic_similarity_with_original(
                user_transcription, original_transcription
            )

            # Harvest the results as they complete.
            for future in as_completed(future_to_score):
                score_name = future_to_score[future]
                try:
                    score_key, score_value = future.result()
                    scores[score_key] = score_value
                except Exception as exc:
                    logging.error(f"{score_name} generated an exception: {exc}")

        logging.info(f"\n\nScores: {scores}")
        return scores

from Scoring.transcription_service import TranscriptionService
from Scoring.karaoke_data import KaraokeData
from Scoring.audio_scorer import AudioScorer
from Scoring.audio_preprocessor import AudioPreprocessor
from typing import List, Dict
import numpy as np
import logging


class Pipeline:
    def __init__(
        self,
        original_audio: np.array,
        track_audio: np.array,
        raw_lyrics_data: str,
        sr: int,
        transcription_method: str,
        score_weights: Dict[str, float],
        pipelines: Dict[str, Dict[str, List[str]]],
        config,
    ):
        self.sr = sr
        self.config = config
        self.pipelines = pipelines
        self.score_weights = score_weights

        # Initialize components
        self.ap = AudioPreprocessor()
        self.audio_scorer = AudioScorer(
            TranscriptionService(method=transcription_method, config=config), "dtaidistance_fast"
        )
        self.karaoke_data = self._initialize_karaoke_data(original_audio, track_audio, raw_lyrics_data, sr)

        # Track scores and chunks
        self._reset_scores()
        self.initialized = False

    def _initialize_karaoke_data(
        self, original_audio: np.array, track_audio: np.array, raw_lyrics_data: str, sr: int
    ) -> KaraokeData:
        """Helper method to initialize the KaraokeData instance."""
        return KaraokeData(original_audio, track_audio, raw_lyrics_data, sr)

    def _reset_scores(self):
        """Reset cumulative scores and chunk count."""
        self.cumulative_scores = {
            "linguistic_accuracy_score": 0,
            "linguistic_similarity_score": 0,
            "amplitude_score": 0,
            "pitch_score": 0,
            "rhythm_score": 0,
        }
        self.chunk_count = 0

    def _preprocess_audio(self, audio: np.array, audio_type: str, **kwargs) -> Dict[str, np.array]:
        """Preprocess audio (either chunk or original) using the specified pipeline."""
        return {
            score_name: self.ap.preprocess_audio(audio, pipeline[audio_type], sr=self.sr, **kwargs)
            for score_name, pipeline in self.pipelines.items()
        }

    def _convert_to_numpy_array(self, audio_chunk, to_float=False):
        dtype = np.int16  # initial dtype, adapt as needed

        # Convert to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=dtype)

        # Convert to floating-point for librosa, if requested
        if to_float:
            audio_array = audio_array.astype(np.float32)
            # Normalize if necessary
            if np.max(np.abs(audio_array)) > 1:
                audio_array = audio_array / np.iinfo(np.int16).max

        return audio_array

    def process_and_score(self, audio_chunk: np.array) -> Dict[str, float]:
        """Process and score a single audio chunk."""
        if not self.initialized:
            self.karaoke_data.align_audio(audio_chunk, method="start")
            self.initialized = True

        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = self._convert_to_numpy_array(audio_chunk, True)

        # Check for non-finite values
        if not np.all(np.isfinite(audio_chunk)):
            logging.error("Audio chunk contains non-finite values!")
            audio_chunk = np.nan_to_num(audio_chunk)
            # Handle accordingly, maybe raise an exception or return

        original_segment, reference_audio = self.karaoke_data.get_next_segment(len(audio_chunk))

        original_segment = self._convert_to_numpy_array(original_segment, True)
        reference_audio = self._convert_to_numpy_array(reference_audio, True)

        # Process audio data
        processed_audio_chunk_data = self._preprocess_audio(audio_chunk, "chunk", reference_audio=reference_audio)
        processed_original_data = self._preprocess_audio(original_segment, "original", reference_audio=reference_audio)

        scores = self._compute_scores(processed_audio_chunk_data, processed_original_data)
        score = self._calculate_weighted_score(scores)
        feedback = self._generate_feedback(scores)

        # Update cumulative scores and chunk count
        for score_name, score_value in scores.items():
            self.cumulative_scores[score_name] += score_value
        self.chunk_count += 1

        average_score = self._calculate_weighted_score(self.cumulative_scores) / self.chunk_count

        return score, average_score, feedback

    def final_score(self):
        average_score = self._calculate_weighted_score(self.cumulative_scores) / self.chunk_count
        feedback = self._generate_feedback(self.cumulative_scores)
        return average_score, feedback

    def _compute_scores(
        self, processed_audio_chunk_data: Dict[str, np.array], processed_original_data: Dict[str, np.array]
    ) -> Dict[str, float]:
        """Compute scores for processed audio data."""
        logging.info(f"\n\n{self.karaoke_data.get_lyrics()}")
        return self.audio_scorer.process_audio_chunk(
            processed_audio_chunk_data, processed_original_data, self.karaoke_data.get_lyrics(), self.sr, False
        )

    def _calculate_weighted_score(self, scores) -> float:
        """Calculates the overall weighted score."""

        weighted_score = 0.0
        for score_type, score_value in scores.items():
            weighted_score += score_value * self.score_weights.get(score_type)

        return weighted_score

    def get_average_scores(self) -> Dict[str, float]:
        """Compute average scores based on processed audio chunks."""
        return {
            score_name: score_value / self.chunk_count for score_name, score_value in self.cumulative_scores.items()
        }

    def _generate_feedback(self, scores: Dict[str, float]) -> str:
        """Generate feedback based on the given scores."""
        feedback_messages = {
            "linguistic_accuracy_score": {
                "low": "🎤 Oops! You might've missed some words or pronounced them differently. Keep practicing the lyrics! 📜",
                "high": "🎤 Great job with the lyrics! You're nailing the words. 🎉",
            },
            "linguistic_similarity_score": {
                "low": "🎤 Hmm, your phrasing seems a bit different from the original. Listen closely to the original singer's style and try to emulate it! 🎶",
                "high": "🎤 You've captured the essence of the original singer's style! Keep it up! 🌟",
            },
            "amplitude_score": {
                "low": "🎤 Your volume seems a bit off. Try to match the song's intensity and dynamics! 📈",
                "high": "🎤 Spot on with the volume! You're in tune with the song's dynamics. 🔊",
            },
            "pitch_score": {
                "low": "🎤 Some notes seem off-pitch. Remember, practice makes perfect! 🎵",
                "high": "🎤 Your pitch is on point! That's some great ear you have there. 🎧",
            },
            "rhythm_score": {
                "low": "🎤 Oops, your timing seems a bit off. Keep practicing to the beat! 🥁",
                "high": "🎤 You've got the rhythm! Great job staying in sync with the beat. 💃",
            },
        }

        # Identify the lowest score and its type
        lowest_score_type = min(scores, key=scores.get)
        lowest_score = scores[lowest_score_type]

        # Provide feedback based on the lowest score if it's less than 0.8
        if lowest_score < 0.8:
            return feedback_messages[lowest_score_type]["low"]
        else:
            # If all scores are above 0.8, provide "high" feedback for the highest score
            highest_score_type = max(scores, key=scores.get)
            return feedback_messages[highest_score_type]["high"]

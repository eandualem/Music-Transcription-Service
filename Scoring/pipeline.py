from transcription_service import TranscriptionService
from karaoke_data import KaraokeData
from audio_scorer import AudioScorer
from audio_preprocessor import AudioPreprocessor
from typing import List, Dict, Union, Tuple, Callable
import numpy as np


class Pipeline:
    def __init__(self,
                 original_audio: np.array,
                 track_audio: np.array,
                 raw_lyrics_data: str,
                 sr: int,
                 transcription_method:str,
                 pipelines: Dict[str, Dict[str, List[str]]]):

        self.sr = sr
        self.pipelines = pipelines

        # Initialize components
        self.ap = AudioPreprocessor()
        self.audio_scorer = AudioScorer(TranscriptionService(method=transcription_method), 'dtaidistance_fast')
        self.karaoke_data = self._initialize_karaoke_data(original_audio, track_audio, raw_lyrics_data, sr)

        # Track scores and chunks
        self._reset_scores()
        self.initialized = False

    def _initialize_karaoke_data(self,
                                 original_audio: np.array,
                                 track_audio: np.array,
                                 raw_lyrics_data: str,
                                 sr: int) -> KaraokeData:
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

    def process_and_score(self, audio_chunk: np.array) -> Dict[str, float]:
        """Process and score a single audio chunk."""
        if not self.initialized:
            self.karaoke_data.align_audio(audio_chunk, method="start")
            self.initialized = True

        original_segment, reference_audio = self.karaoke_data.get_next_segment(len(audio_chunk))

        # Process audio data
        processed_audio_chunk_data = self._preprocess_audio(audio_chunk, "chunk", reference_audio=reference_audio)
        processed_original_data = self._preprocess_audio(original_segment, "original", reference_audio=reference_audio)

        scores = self._compute_scores(processed_audio_chunk_data, processed_original_data)
        feedback = self._generate_feedback(scores)

        # Update cumulative scores and chunk count
        for score_name, score_value in scores.items():
            self.cumulative_scores[score_name] += score_value
        self.chunk_count += 1

        return scores, feedback

    def _compute_scores(self,
                        processed_audio_chunk_data: Dict[str, np.array],
                        processed_original_data: Dict[str, np.array]) -> Dict[str, float]:
        """Compute scores for processed audio data."""
        return self.audio_scorer.process_audio_chunk(
            processed_audio_chunk_data,
            processed_original_data,
            self.karaoke_data.get_lyrics(),
            self.sr,
            True
        )

    def get_average_scores(self) -> Dict[str, float]:
        """Compute average scores based on processed audio chunks."""
        return {
            score_name: score_value / self.chunk_count
            for score_name, score_value in self.cumulative_scores.items()
        }

    def _generate_feedback(self, scores: Dict[str, float]) -> str:
        """Generate feedback based on the given scores."""
        feedback_messages = {
            "linguistic_accuracy_score": {
                "low": "🎤 Oops! You might've missed some words or pronounced them differently. Keep practicing the lyrics! 📜",
                "high": "🎤 Great job with the lyrics! You're nailing the words. 🎉"
            },
            "linguistic_similarity_score": {
                "low": "🎤 Hmm, your phrasing seems a bit different from the original. Listen closely to the original singer's style and try to emulate it! 🎶",
                "high": "🎤 You've captured the essence of the original singer's style! Keep it up! 🌟"
            },
            "amplitude_score": {
                "low": "🎤 Your volume seems a bit off. Try to match the song's intensity and dynamics! 📈",
                "high": "🎤 Spot on with the volume! You're in tune with the song's dynamics. 🔊"
            },
            "pitch_score": {
                "low": "🎤 Some notes seem off-pitch. Remember, practice makes perfect! 🎵",
                "high": "🎤 Your pitch is on point! That's some great ear you have there. 🎧"
            },
            "rhythm_score": {
                "low": "🎤 Oops, your timing seems a bit off. Keep practicing to the beat! 🥁",
                "high": "🎤 You've got the rhythm! Great job staying in sync with the beat. 💃"
            }
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

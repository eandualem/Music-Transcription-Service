from karaoke_data import KaraokeData
from audio_scorer import AudioScorer
from audio_preprocessor import AudioPreprocessor
from google_speech import GoogleSpeechTranscription
from typing import List, Dict, Union, Tuple, Callable

class Pipeline:
    def __init__(self,
                 original_audio: np.array,
                 track_audio: np.array,
                 raw_lyrics_data: str,
                 sr: int,
                 pipelines: Dict[str, Dict[str, List[str]]]):

        self.sr = sr
        self.pipelines = pipelines

        # Initialize components
        self.ap = AudioPreprocessor()
        self.audio_scorer = AudioScorer(GoogleSpeechTranscription(), 'dtaidistance_fast')
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

    def _preprocess_audio(self, audio: np.array, audio_type: str) -> Dict[str, np.array]:
        """
        Preprocess audio (either chunk or original) using the specified pipeline.

        Parameters:
        - audio (np.array): The audio data to preprocess.
        - audio_type (str): The type of audio, either "chunk" or "original".

        Returns:
        - Dict[str, np.array]: A dictionary of preprocessed audio data for each score.
        """
        return {
            score_name: self.ap.preprocess_audio(audio, pipeline[audio_type])
            for score_name, pipeline in self.pipelines.items()
        }

    def process_and_score(self, audio_chunk: np.array) -> Dict[str, float]:
        """Process and score a single audio chunk."""
        if not self.initialized:
            self.karaoke_data.align_audio(audio_chunk, method="start")
            self.initialized = True

        original_segment, _ = self.karaoke_data.get_next_segment(len(audio_chunk))

        # Process audio data
        processed_audio_chunk_data = self._preprocess_audio(audio_chunk, "chunk")
        processed_original_data = self._preprocess_audio(original_segment, "original")

        scores = self._compute_scores(processed_audio_chunk_data, processed_original_data)

        # Update cumulative scores and chunk count
        for score_name, score_value in scores.items():
            self.cumulative_scores[score_name] += score_value
        self.chunk_count += 1

        return scores

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

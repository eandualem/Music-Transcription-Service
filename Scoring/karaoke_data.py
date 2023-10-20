import librosa
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple


class KaraokeData:
    """Manages data and alignment for karaoke songs."""

    # Constants for alignment methods
    ALIGNMENT_METHODS = {
        "onset_detection": "_align_onset_detection",
        "lyrics_data": "_align_lyrics_data",
        "start": "_align_start",
    }

    def __init__(self, original_audio: np.array, track_audio: np.array, raw_lyrics_data: str, sampling_rate: int):
        """Initializes the karaoke data object with audio and lyrics."""
        self.original_audio = original_audio
        self.track_audio = track_audio
        self.sampling_rate = sampling_rate
        self.current_position = 0
        self.previous_position = 0
        self.initial_alignment_done = False
        self.lyrics_data = self._parse_lyrics(raw_lyrics_data)

    def get_lyrics(self, start_time: float = None, end_time: float = None) -> List[str]:
        """Fetches the lyrics between the specified start and end times."""

        start_time = start_time or librosa.samples_to_time(self.previous_position, sr=self.sampling_rate)
        end_time = end_time or librosa.samples_to_time(self.current_position, sr=self.sampling_rate)

        lyrics_within_interval = []
        for entry in self.lyrics_data:
            if start_time <= entry["time"] <= end_time:
                lyrics_within_interval.append(entry["lyrics"])
        return self._transform_lyrics(lyrics_within_interval)

    def _transform_lyrics(self, lyrics_list):
        transformed_lyrics = ""
        word = ""
        for syllable in lyrics_list:
            if "\\n" in syllable:
                parts = syllable.split("\\n")
                word += parts[0]
                transformed_lyrics += word.strip() + "\n"
                word = parts[1] + " "
            else:
                word += syllable + " "
        transformed_lyrics += word.strip()
        return transformed_lyrics

    def _parse_lyrics(self, raw_lyrics: str) -> List[Dict[str, Union[float, str]]]:
        """Converts raw CSV lyrics into a structured format."""
        try:
            # Load the CSV data
            csv_data = pd.read_csv(raw_lyrics)
            csv_data = csv_data[csv_data["payload_type"] == 1]
            parsed_lyrics = [
                {"time": row["start_time"], "lyrics": row["payload"].strip()} for _, row in csv_data.iterrows()
            ]

        except Exception as e:
            logging.error(f"Error parsing CSV lyrics: {e}")
            parsed_lyrics = []

        return parsed_lyrics

    def align_audio(self, audio_chunk: np.array, method: str = "cross_correlation"):
        """Aligns the audio using the specified method."""
        if method not in self.ALIGNMENT_METHODS:
            raise ValueError(f"Invalid alignment method. Available methods: {list(self.ALIGNMENT_METHODS.keys())}")
        alignment_method = getattr(self, self.ALIGNMENT_METHODS[method])
        return alignment_method(audio_chunk)

    def get_next_segment(self, audio_chunk_length: int) -> Tuple[np.array, np.array]:
        """
        Retrieve the next segment from the original audio and track audio based on the provided chunk length.
        """
        if not self.initial_alignment_done:
            raise ValueError("Initial alignment is required before accessing subsequent segments.")
        end_sample = self.current_position + audio_chunk_length
        original_segment = self.original_audio[self.current_position : min(end_sample, len(self.original_audio))]
        track_segment = self.track_audio[self.current_position : min(end_sample, len(self.track_audio))]
        self.previous_position = self.current_position
        self.current_position = min(end_sample, len(self.original_audio))
        return original_segment, track_segment

    def _get_audio_segment(self, audio: np.array, segment_length: float) -> np.array:
        """Extracts a segment of audio based on current position and segment length."""
        start_sample = librosa.time_to_samples(self.current_position, sr=self.sampling_rate)
        end_sample = start_sample + librosa.time_to_samples(segment_length, sr=self.sampling_rate)
        return audio[start_sample:end_sample]

    def _align_start(self, audio_chunk: np.array):
        """Aligns the audio starting at the beginning."""
        self.current_position = 0
        self.initial_alignment_done = True

    def _align_lyrics_data(self, audio_chunk: np.array):
        """Aligns the audio using the first entry in lyrics data."""
        if self.lyrics_data and not self.initial_alignment_done:
            start_time = self.lyrics_data[0]["time"]
            self.current_position = librosa.time_to_samples(start_time, sr=self.sampling_rate)
            self.initial_alignment_done = True

    def _align_onset_detection(self, audio_chunk: np.array) -> int:
        """Aligns the audio using onset detection and returns the onset position in the audio chunk."""
        onset_position_in_chunk = 0
        if not self.initial_alignment_done:
            try:
                original_onsets = librosa.onset.onset_detect(y=self.original_audio, sr=self.sampling_rate)
                chunk_onsets = librosa.onset.onset_detect(y=audio_chunk, sr=self.sampling_rate)
                original_onset_samples = librosa.frames_to_samples(original_onsets)
                chunk_onset_samples = librosa.frames_to_samples(chunk_onsets)
                if original_onset_samples.size > 0 and chunk_onset_samples.size > 0:
                    offset = original_onset_samples[0] - chunk_onset_samples[0]
                    self.current_position = max(0, offset)
                    onset_position_in_chunk = chunk_onset_samples[0]
                else:
                    self.current_position = 0
                self.initial_alignment_done = True
            except Exception as e:
                logging.error(f"Error in onset detection alignment: {e}")
        return onset_position_in_chunk

    def reset_alignment(self):
        """Resets the alignment state."""
        self.current_position = 0
        self.initial_alignment_done = False

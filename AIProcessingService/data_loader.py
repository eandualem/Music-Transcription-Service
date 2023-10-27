import io
import requests
import pandas as pd
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor


class DataLoader:
    def __init__(self, lyrics_url, track_url, voice_helper_url):
        self.session = requests.Session()  # Create a session object for reusing TCP connections

        # Utilize ThreadPoolExecutor to perform download and resample operations concurrently
        with ThreadPoolExecutor() as executor:
            self.lyrics_data_future = executor.submit(self._download_and_process_lyrics, lyrics_url)
            self.original_audio_future = executor.submit(self._download_and_resample, track_url)
            self.background_track_future = executor.submit(self._download_and_resample, voice_helper_url)

        # Get the results from the futures
        self.lyrics_data = self.lyrics_data_future.result()
        self.original_audio = self.original_audio_future.result()
        self.background_track = self.background_track_future.result()

    def _download_data_from_url(self, url):
        """Download data from a given URL."""
        response = self.session.get(url)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to download data from URL: {url}")

    def _download_and_process_lyrics(self, url):
        """Download and process lyrics data."""
        lyrics_bytes = self._download_data_from_url(url)
        lyrics_file_like = io.BytesIO(lyrics_bytes)
        return pd.read_csv(lyrics_file_like)

    def _resample_audio(self, audio_bytes):
        """Resample audio data to 8000 Hz."""
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        resampled_audio_segment = audio_segment.set_frame_rate(8000)
        return resampled_audio_segment.raw_data

    def _download_and_resample(self, url):
        """Download and resample audio data concurrently."""
        audio_bytes = self._download_data_from_url(url)
        # Now directly call the resample_audio function without creating a new thread
        return self._resample_audio(audio_bytes)

    def get_data(self):
        return self.lyrics_data, self.original_audio, self.background_track

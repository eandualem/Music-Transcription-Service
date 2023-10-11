import re
import requests

class AudioLoader:
    def __init__(self, lyrics_url, track_url, voice_helper_url):
        self.lyrics_url = lyrics_url
        self.track_url = track_url
        self.voice_helper_url = voice_helper_url
        # Attributes to store downloaded data
        self.lyrics_data = None
        self.original_audio = None
        self.background_track = None

    def download_lyrics(self):
        """Download and parse LRC lyrics."""
        lyrics_raw = self._download_data_from_url(self.lyrics_url)
        self.lyrics_data = self._parse_lrc(lyrics_raw.decode('utf-8'))

    def download_audio(self, url):
        """Download audio data."""
        audio_data = self._download_data_from_url(url)
        return audio_data

    def _download_data_from_url(self, url):
        """Download data from a given URL."""
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to download data from URL: {url}")

    def _parse_lrc(self, lrc_content):
        """Parse LRC format to extract time and text."""
        # Extract timestamps and text using regex
        pattern = re.compile(r'\[(\d{2}:\d{2}\.\d{2})\](.*)')
        matches = pattern.findall(lrc_content)

        parsed_lyrics = []

        for time_stamp, text in matches:
            minutes, seconds = time_stamp.split(":")
            total_seconds = int(minutes) * 60 + float(seconds)

            # Only add entries with non-empty text
            if text.strip():
                parsed_lyrics.append({"time": total_seconds, "text": text.strip()})

        return parsed_lyrics

    def load_original_audio(self):
        self.original_audio = self.download_audio(self.track_url)

    def load_background_track(self):
        self.background_track = self.download_audio(self.voice_helper_download_url)

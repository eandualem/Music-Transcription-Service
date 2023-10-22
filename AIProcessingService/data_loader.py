import requests


class DataLoader:
    def __init__(self, lyrics_url, track_url, voice_helper_url):
        self.lyrics_data = self._download_data_from_url(lyrics_url)
        self.original_audio = self._download_data_from_url(track_url)
        self.background_track = self._download_data_from_url(voice_helper_url)

    def _download_data_from_url(self, url):
        """Download data from a given URL."""
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to download data from URL: {url}")

    def get_data(self):
        return self.lyrics_data, self.original_audio, self.background_track

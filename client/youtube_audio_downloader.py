import logging
import yt_dlp
import os

logging.basicConfig(level=logging.INFO)


class YouTubeAudioDownloader:

    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'logger': logging,
            'progress_hooks': [self.hook],
            'outtmpl': 'downloaded_audio.%(ext)s'  # specify the name template for downloaded files
        }

    def hook(self, d):
        if d['status'] == 'finished':
            self.file_path = d['filename']
            logging.info(f"Downloaded audio successfully to: {d['filename']}")

    def download_audio(self, youtube_url):
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([youtube_url])
        return os.path.abspath(self.file_path)

import logging
from audio_slicer import AudioSlicer
from youtube_audio_downloader import YouTubeAudioDownloader
from audio_transcription_client import AudioTranscriptionClient

# Setting up basic logging configuration
logging.basicConfig(level=logging.INFO)


class Client:
    def __init__(self,):
        self.downloader = YouTubeAudioDownloader()
        self.slicer = AudioSlicer()
        self.client = AudioTranscriptionClient()

    def run(self):
        link = 'https://www.youtube.com/watch?v=D2_K_Oa_QaI'
        audio_file = self.downloader.download_audio(link)
        sliced_audio_file = self.slicer.slice_audio(audio_file, 18, 52)

        whisper_trans = self.client.send_audio_to_server(sliced_audio_file, 'whisper')
        print("Whisper transcription:", whisper_trans)

        whisper_trans = self.client.send_audio_to_server(sliced_audio_file, 'google_speech')
        print("Google speech transcription:", whisper_trans)

        whisper_trans = self.client.send_audio_to_server(sliced_audio_file, 'wav2vec2')
        print("Wav2vec2 transcription:", whisper_trans)


if __name__ == '__main__':
    client = Client()
    client.run()

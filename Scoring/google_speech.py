import io
import librosa
import logging
import numpy as np
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)

class GoogleSpeechTranscription:
    def __init__(self):
        # Initialize the Google Speech client
        client_file = "sa_speech_test.json"
        credentials = service_account.Credentials.from_service_account_file(client_file)
        self.client = speech.SpeechClient(credentials=credentials)

    def transcribe(self, audio_data, sr=None, from_file=False):
        # Convert audio data if it's coming from a file (gRPC connection)
        if not from_file:
            audio_bytes_io = io.BytesIO(audio_data)
            audio_array, sr = librosa.load(audio_bytes_io, sr=None)
            audio_content = np.int16(audio_array * 32767).tobytes()
        else:
            audio_content = np.int16(audio_data * 32767).tobytes()

        # Prepare the audio and config objects for the API request
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sr,
            language_code="en-US",
            model="video",
        )

        # Make the API request
        try:
            response = self.client.recognize(config=config, audio=audio)
            if response.results:
                transcription = response.results[0].alternatives[0].transcript
                return transcription
            else:
                logging.warning("No transcription results returned from Google Speech API.")
                return ""
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return ""

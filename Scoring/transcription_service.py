import openai
import logging
from google.oauth2 import service_account
from google.cloud import speech_v1 as speech

logging.basicConfig(level=logging.INFO)


class Transcription:
    def transcribe(self, wav_file_path: str) -> str:
        raise NotImplementedError


class GoogleSpeechTranscription(Transcription):
    def __init__(self):
        client_file = "./sa_speech_test.json"
        credentials = service_account.Credentials.from_service_account_file(client_file)
        self.client = speech.SpeechClient(credentials=credentials)

    def transcribe(self, wav_file_path: str) -> str:
        # Read the audio file
        with open(wav_file_path, "rb") as audio_file:
            audio_content = audio_file.read()

        # Prepare the audio and config objects for the API request
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,
            language_code="en-US",
            model="video",
        )

        # Make the API request
        try:
            response = self.client.recognize(config=config, audio=audio)
            # logging.info(f"Google API Response: {response}")

            if response.results:
                transcription = None
                for result in response.results:
                    for alternative in result.alternatives:
                        if alternative.transcript:
                            transcription = alternative.transcript
                            break  # Exit inner loop once a transcript is found
                    if transcription:
                        break  # Exit outer loop if a transcript was found in inner loop

            else:
                logging.warning("No transcription results returned from Google Speech API.")
                return ""
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return ""


class WhisperSpeechTranscription(Transcription):
    def __init__(self, api_key):
        openai.api_key = api_key

    def transcribe(self, wav_file_path: str) -> str:
        """Transcribe the provided audio data using the Whisper API."""
        try:
            # Transcribe the audio file
            with open(wav_file_path, "rb") as f:
                response = openai.Audio.transcribe("whisper-1", file=f)

            # Check for transcription text in the response
            if response.text:
                transcription = response.text
            else:
                logging.warning("No transcription results returned from Whisper API. ⚠️")
                transcription = ""
        except Exception as e:
            logging.error(f"Error transcribing audio: {e} 🚫")
            transcription = ""

        return transcription


class TranscriptionService:
    def __init__(self, method: str, config):
        self.strategy = None
        if method == "google":
            self.strategy = GoogleSpeechTranscription()
        elif method == "whisper":
            self.strategy = WhisperSpeechTranscription(config.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported transcription method: {method}")

    def transcribe(self, wav_file_path: str) -> str:
        return self.strategy.transcribe(wav_file_path)

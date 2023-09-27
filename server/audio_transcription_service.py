import logging
from generated import audio_transcription_pb2
from generated import audio_transcription_pb2_grpc
from whisper_transcription import WhisperTranscription
from wav2vec2_transcription import Wav2VecTranscription
from google_speech_transcription import GoogleSpeechTranscription

logging.basicConfig(level=logging.INFO)


class AudioTranscriptionService(audio_transcription_pb2_grpc.AudioTranscriptionServiceServicer):
    def __init__(self):
        self.whisper_transcriber = WhisperTranscription()
        logging.info("OpenAI Whisper loaded successfully")
        self.google_speech_transcriber = GoogleSpeechTranscription()
        logging.info("Google speech to text loaded successfully")
        self.wav2vec2_transcriber = Wav2VecTranscription()
        logging.info("Facebook Wav2Vec2 loaded successfully")

    def TranscribeAudio(self, request, context):
        transcription_service = request.transcription_service

        if transcription_service == "whisper":
            transcription_text = self.whisper_transcriber.transcribe(request.audio_data)
        elif transcription_service == "google_speech":
            transcription_text = self.google_speech_transcriber.transcribe(request.audio_data)
        elif transcription_service == "wav2vec2":
            transcription_text = self.wav2vec2_transcriber.transcribe(request.audio_data)
        else:
            logging.error(f"Invalid transcription service: {transcription_service}")
            # Handle the error or set a default transcription service

        return audio_transcription_pb2.TranscriptionResponse(transcription=transcription_text)

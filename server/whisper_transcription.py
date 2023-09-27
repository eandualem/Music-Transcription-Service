import io
import whisper
import librosa
import logging 

logging.basicConfig(level=logging.INFO)


class WhisperTranscription:
    def __init__(self):
        self.model = whisper.load_model("large-v2")

    def transcribe(self, audio_data):
        audio_bytes_io = io.BytesIO(audio_data)
        audio_array, _ = librosa.load(audio_bytes_io, sr=None)
        transcription = self.model.transcribe(audio_array, verbose=True)
        return transcription['text']

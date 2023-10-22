import os
import openai
import logging
import numpy as np
import tempfile
import scipy.io.wavfile as wav


class WhisperSpeechTranscription:
    def __init__(self,):
        """
        Initialize the OpenAI client with the provided API key.
        """
        openai.api_key = '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'

    def transcribe(self, audio_data: np.ndarray, sr: int, from_file=False) -> str:
        """
        Transcribe the provided audio data using the Whisper API.
        """

        # Create a temporary file to store the audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav.write(temp_file.name, sr, audio_data)

        try:
            # Transcribe the audio file
            with open(temp_file.name, "rb") as f:
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
        finally:
            # Ensure temporary file is deleted
            os.remove(temp_file.name)

        return transcription

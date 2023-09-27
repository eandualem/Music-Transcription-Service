import torch
import logging
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

logging.basicConfig(level=logging.INFO)


class Wav2VecTranscription:
    def __init__(self, device="cpu"):
        # Check if GPU is available and set the device
        if torch.cuda.is_available():
            device = "cuda"
        self.device = device
        
        # Load the Wav2Vec 2.0 model and tokenizer
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    def transcribe(self, audio_path):
        try:
            # Load the audio data
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample the audio to 16kHz if it's not already
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # Tokenize the audio data
            input_values = self.tokenizer(waveform.squeeze().numpy(), return_tensors="pt", padding="longest").input_values

            # Use the Wav2Vec 2.0 model to transcribe the audio data
            with torch.no_grad():
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)

            # Decode the ids to text
            transcription = self.tokenizer.batch_decode(predicted_ids)[0]
            return transcription
        
        except Exception as e:
            print(f"Error: {e}")
            return None

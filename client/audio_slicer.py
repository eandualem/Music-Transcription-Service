import os
import librosa
import logging
import subprocess
import soundfile as sf

# Setting up basic logging configuration
logging.basicConfig(level=logging.INFO)


class AudioSlicer:
    def __init__(self,):
        pass

    def slice_audio(self, audio_file_path, start_time, end_time):
        # Convert the audio file to WAV format using ffmpeg
        logging.info(f"audio_file_path_1: {audio_file_path}")
        wav_file_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
        subprocess.run(['ffmpeg', '-i', audio_file_path, wav_file_path])

        # Load the WAV audio file using librosa
        y, sr = librosa.load(wav_file_path, sr=None)

        # Calculate the start and end sample indices
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)

        # Slice the audio numpy array
        y_sliced = y[start_sample:end_sample]

        # Save the sliced audio to a WAV file
        output_file_path = "sliced_audio.wav"
        sf.write(output_file_path, y_sliced, sr)

        # Clean up: Remove the original downloaded file and the converted WAV file
        logging.info(f"audio_file_path: {audio_file_path}")
        logging.info(f"wav_file_path: {wav_file_path}")
        logging.info(f"output_file_path: {output_file_path}")

        # os.remove(audio_file_path)
        os.remove(wav_file_path)
        logging.info(f"Sliced audio saved to: {output_file_path}")

        return output_file_path

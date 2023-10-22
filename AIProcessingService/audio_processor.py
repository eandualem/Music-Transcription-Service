from logger import Logger
from google.protobuf.duration_pb2 import Duration
from Declarations.Model.AIProcessingService import AIProcessingResponse_pb2
from audio_loader import AudioLoader
import numpy as np
import librosa

class AudioProcessor:
    def __init__(self, client_token, initial_data):
        self.client_token = client_token
        self.log = Logger.get_logger(__name__)

        # Initialize modular components
        self.audio_loader = AudioLoader(initial_data.lyrics_download_url, initial_data.track_download_url, initial_data.voice_helper_download_url)
        self.audio_scorer = AudioScorer(self.audio_loader.original_audio)

        # To keep track of the total processed audio duration
        self.processed_duration = 0
        self.chunk_scores = []

    def create_status_response(self, status_code):
        """Creates an AI processing response containing only a status code.

        Args:
            status_code (AIProcessingResponse_pb2.AIProcessingResponse.StatusCode): The status code to be set.

        Returns:
            AIProcessingResponse: The constructed response with only a status code.
        """

        response = AIProcessingResponse_pb2.AIProcessingResponse()
        response.status_code = status_code
        return response

    # def process_audio_chunk(self, request):
    #     """Processes an audio chunk and returns the corresponding AI processing response.

    #     Args:
    #         request: The audio chunk request containing audio data.

    #     Returns:
    #         AIProcessingResponse: The response with processing results.
    #     """
    #     transcription_text = self.google_speech_transcriber.transcribe(request.audio_chunk.audio_data)
    #     self.log.info(f"Transcription text: {transcription_text}")
    #     # TODO: Implement actual processing
    #     return self._create_processing_response(1.95, 1.9, 20)

    def process_audio_chunk(self, request):
        user_audio_chunk = request.audio_chunk.audio_data
        self.log.debug(f"Received audio chunk of length {len(user_audio_chunk)}")

        # Align user's audio chunk with the original
        aligned_user_audio_chunk = AudioUtils.align_audio(user_audio_chunk, self.audio_loader.original_audio)
        self.log.debug(f"Aligned audio chunk of length {len(aligned_user_audio_chunk)}")

        # Score using amplitude, spectral, and MFCC matching
        amplitude_score = self.audio_scorer.amplitude_matching_score(aligned_user_audio_chunk, self.audio_loader.original_audio)
        self.log.debug(f"Amplitude score: {amplitude_score}")
        spectral_score = self.audio_scorer.spectral_matching_score(aligned_user_audio_chunk, self.audio_loader.original_audio)
        self.log.debug(f"Spectral score: {spectral_score}")
        mfcc_score = self.audio_scorer.mfcc_matching_score(aligned_user_audio_chunk, self.audio_loader.original_audio)
        self.log.debug(f"MFCC score: {mfcc_score}")

        # Compute combined score using weights
        combined_score = self.audio_scorer.combined_score(amplitude_score, spectral_score, mfcc_score)
        self.log.debug(f"Combined score: {combined_score}")
        self.chunk_scores.append(combined_score)

        # Compute average score
        average_score = np.mean(self.chunk_scores)

        # The instant score can be the score of the current chunk
        instant_score = combined_score

        # Generate feedback
        feedback = self.generate_feedback(amplitude_score, spectral_score, mfcc_score)
        self.log.debug(f"Feedback: {feedback}")

        # Construct the response and return
        response = self._create_processing_response(instant_score, average_score, len(user_audio_chunk) / librosa.get_samplerate(request.audio_chunk.audio_data))
        response.feedback = feedback
        return response

    def generate_feedback(self, amplitude_score, spectral_score, mfcc_score):
        """Generate feedback based on individual scoring metrics."""
        feedback = []

        # Amplitude feedback
        if amplitude_score < 0.5:
            feedback.append("Your volume or energy level seems different from the original. Try to match the song's intensity.")

        # Spectral feedback
        if spectral_score < 0.5:
            feedback.append("Your pitch or tone might be off. Practice matching the original singer's pitch.")

        # MFCC feedback
        if mfcc_score < 0.5:
            feedback.append("The overall sound quality or timbre of your voice seems different. Try to match the original singer's voice characteristics.")

        return ' '.join(feedback)

    def _create_processing_response(self, instant_score, average_score, seconds):
        """Creates an AI processing response based on provided parameters.

        Args:
            instant_score (float): The instant score for the audio chunk.
            average_score (float): The average score for the audio chunk.
            seconds (int): The processed duration in seconds.

        Returns:
            AIProcessingResponse: The constructed response.
        """

        self.log.info("Creating AIProcessingResponse AudioChunk")
        review = AIProcessingResponse_pb2.AIProcessingResponse()
        review.live_review.instant_score = instant_score
        review.live_review.average_score = average_score
        duration = Duration(seconds=seconds)
        review.live_review.processed_duration.CopyFrom(duration)
        return AIProcessingResponse_pb2.AIProcessingResponse(live_review=review.live_review)

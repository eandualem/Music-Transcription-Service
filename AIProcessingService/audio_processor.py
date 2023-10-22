import librosa
from logger import Logger
from Scoring.pipeline import Pipeline
from Declarations.Model.Report_pb2 import Report
from google.protobuf.duration_pb2 import Duration
from AIProcessingService.data_loader import DataLoader
from Declarations.Model.AIProcessingService import AIProcessingResponse_pb2

score_weights = {
    "linguistic_accuracy_score": 0.2,
    "linguistic_similarity_score": 0.2,
    "amplitude_score": 0.2,
    "pitch_score": 0.2,
    "rhythm_score": 0.2,
}

pipelines = {
    "linguistic_accuracy_score": {"chunk": [], "original": []},
    "linguistic_similarity_score": {"chunk": [], "original": []},
    "amplitude_score": {"chunk": [], "original": []},
    "pitch_score": {
        "chunk": ["adaptive_noise_reduction", "spectral_gate", "normalize"],
        "original": ["spectral_gate", "normalize"],
    },
    "rhythm_score": {
        "chunk": ["adaptive_noise_reduction", "spectral_gate", "normalize"],
        "original": ["spectral_gate", "normalize"],
    },
}


class AudioProcessor:
    def __init__(self, client_token, private_interface_client):
        self.client_token = client_token
        self.log = Logger.get_logger(__name__)
        self.private_interface_client = private_interface_client
        initial_data = self.private_interface_client.fetch_initial_data(self.client_token)

        # Initialize modular components
        data_loader = DataLoader(
            initial_data.lyrics_download_url, initial_data.track_download_url, initial_data.voice_helper_download_url
        )
        lyrics_data, original_audio, background_track = data_loader.get_data()
        self.pipeline = Pipeline(
            original_audio=original_audio,
            track_audio=background_track,
            raw_lyrics_data=lyrics_data,
            sr=44100,
            transcription_method="whisper",
            score_weights=score_weights,
            pipelines=pipelines,
        )

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

    def process_audio_chunk(self, request):
        user_audio_chunk = request.audio_chunk.audio_data
        self.log.debug(f"Received audio chunk of length {len(user_audio_chunk)}")
        score, average_score, feedback = self.pipeline.process_and_score(user_audio_chunk)

        # Construct the response and return
        response = self._create_processing_response(
            score, average_score, len(user_audio_chunk) / librosa.get_samplerate(request.audio_chunk.audio_data)
        )
        response.feedback = feedback
        return response

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

    def handle_finalize(self, request):
        """Handles the finalize request, generates a report, and sends it to the private interface client.

        Args:
            request: The finalize request.
            client_token: The client token used for authentication.

        Returns:
            AIProcessingResponse: The constructed response containing the generated report.
        """
        self.log.info(f"Handling Finalize request with reason: {request.finalize.finalize_reason}")
        average_score, feedback = self.pipeline.final_score()

        report = self._create_report_feedback(average_score, feedback, 20, 20, 2.9)
        self.private_interface_client.report_processing_result(report, self.client_token)
        return AIProcessingResponse_pb2.AIProcessingResponse(report=report)

    def _create_report_feedback(self, avg_score, ai_comment, start_sec, end_sec, report_avg_score):
        """Creates a report feedback based on the provided parameters.

        Args:
            avg_score (float): The average score for the report.
            ai_comment (str): The AI-generated comment for the report.
            start_sec (int): The start time (in seconds) of the report segment.
            end_sec (int): The end time (in seconds) of the report segment.
            report_avg_score (float): The average score for the entire report.

        Returns:
            Report: The constructed report feedback.
        """
        report = Report()
        if not report.feedback:
            report.feedback.add()
        feedback_item = report.feedback[0]
        feedback_item.average_score = avg_score
        feedback_item.ai_comment = ai_comment
        feedback_item.start_time.CopyFrom(Duration(seconds=start_sec))
        feedback_item.end_time.CopyFrom(Duration(seconds=end_sec))
        report.average_score = report_avg_score
        return report

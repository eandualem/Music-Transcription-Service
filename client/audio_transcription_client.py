import grpc
import logging
from generated import audio_transcription_pb2
from generated import audio_transcription_pb2_grpc

# Setting up basic logging configuration
logging.basicConfig(level=logging.INFO)


class AudioTranscriptionClient:
    def __init__(self, server_address='grpc-server:50051'):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = audio_transcription_pb2_grpc.AudioTranscriptionServiceStub(self.channel)

    def send_audio_to_server(self, audio_file, trans_service):
        with open(audio_file, 'rb') as f:
            audio_data = f.read()

        response = self.stub.TranscribeAudio(audio_transcription_pb2.AudioRequest(audio_data=audio_data, transcription_service=trans_service))
        logging.info("Received transcription from the server.")
        return response.transcription

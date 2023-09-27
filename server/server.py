import grpc
import logging
from concurrent import futures
from generated import audio_transcription_pb2_grpc
from audio_transcription_service import AudioTranscriptionService


# Setting up basic logging configuration
logging.basicConfig(level=logging.INFO)


class Server:
    def __init__(self, port=50051):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self.port = port

    def start(self):
        audio_transcription_pb2_grpc.add_AudioTranscriptionServiceServicer_to_server(AudioTranscriptionService(), self.server)
        self.server.add_insecure_port(f'[::]:{self.port}')
        self.server.start()
        logging.info(f"Server started and listening on port {self.port}.")
        self.server.wait_for_termination()


if __name__ == '__main__':
    server = Server()
    server.start()

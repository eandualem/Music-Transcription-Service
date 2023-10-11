import grpc
import time
from logger import Logger
from request_builder import RequestBuilder
from ai_processing_service_client import AIProcessingServiceClient

# Constants for configuration
USER_LOCALE = "en"
AUTH_TOKEN = "token_placeholder"
CERT_FILE_PATH = "certificate.pem"
SERVER_ADDRESS = "aiprocessing-service:50051"


class Client:
    def __init__(self):
        self.client_token = AUTH_TOKEN
        self.user_locale = USER_LOCALE
        self.client = AIProcessingServiceClient(server_address=SERVER_ADDRESS, cert_file=CERT_FILE_PATH)
        self.log = Logger.get_logger(__name__)

    def read_audio_data_from_file(self, file_path):
        """Read raw audio data from a file."""
        with open(file_path, "rb") as f:
            return f.read()

    def build_requests(self):
        """Build requests for processing."""
        # Create Initialize request
        initialize_request = RequestBuilder.create_initialize_request(self.client_token)

        # Create AudioChunk requests
        audio_files = ["part1.wav", "part2.wav", "part3.wav"]
        audio_chunk_requests = [
            RequestBuilder.create_audio_chunk_request(self.read_audio_data_from_file(file)) for file in audio_files
        ]

        # Create Finalize request
        finalize_request = RequestBuilder.create_finalize_request(0)

        return [initialize_request] + [audio_chunk_requests[0]] + [finalize_request]

    def process_requests(self, requests):
        """Process the provided requests and log responses."""
        self.log.info("Processing requests...")
        for response in self.client.process(requests, self.client_token, self.user_locale):
            self.log.info(f"Received response: {response}")
            time.sleep(20)

    def run(self):
        try:
            self.log.info("Initializing AIProcessingServiceClient...")
            requests = self.build_requests()
            self.process_requests(requests)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                self.log.error("Authentication failed with the AI Processing Service.")
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                self.log.error("AI Processing Service is currently unavailable.")
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                self.log.error("Request to the AI Processing Service timed out.")
            else:
                self.log.error(f"gRPC error occurred: {e.details()}")
        except Exception as e:
            self.log.error(f"An unexpected error occurred: {type(e).__name__} - {e}")

        self.log.info("Client run completed.")


if __name__ == "__main__":
    client = Client()
    client.run()

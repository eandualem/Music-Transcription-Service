import grpc
from logger import Logger
from token_validator import TokenValidator
from audio_processor import AudioProcessor
from error_response import ErrorResponse


class GRPCRequestHandler:
    def __init__(self, private_interface_client):
        self.log = Logger.get_logger(__name__)
        self.audio_processor = None
        self.private_interface_client = private_interface_client

    def handle_initialize_request(self, request):
        """Handles an initialization request and performs necessary actions.

        Args:
            request: The initialization request containing client data.

        Returns:
            AIProcessingResponse: The response indicating success or error.
        """

        try:
            client_token_container = request.initialize.client_token
        except Exception as e:
            self.log.error(f"Error handling initialize request: {str(e)}")
            return ErrorResponse.generate_error_response(
                code=grpc.StatusCode.INTERNAL, details="An error occurred while processing your request."
            )

        if client_token_container:
            client_token = TokenValidator.unpack_client_token(client_token_container)

            self.audio_processor = AudioProcessor(client_token, self.private_interface_client)
            self.log.info("Processing audio chunk...")

            yield self.audio_processor.create_status_response(0)
        else:
            self.log.error("client_token not found in Initialize message.")
            return None

    def handle_audio_chunk_request(self, request):
        yield self.audio_processor.process_audio_chunk(request)

    def handle_finalize_request(self, request):
        yield self.audio_processor.handle_finalize(request)

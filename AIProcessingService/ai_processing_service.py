from logger import Logger
from metadata_utils import MetadataUtils
from Declarations.Service import AIProcessingService_pb2_grpc
from ai_private_interface_service_client import AIPrivateInterfaceServiceClient
from grpc_request_handler import GRPCRequestHandler
from error_response import ErrorResponse


class AIProcessingService(AIProcessingService_pb2_grpc.AIProcessingServiceServicer):
    def __init__(self, server_address, cert_file):
        self.log = Logger.get_logger(__name__)
        self.request_handler = GRPCRequestHandler(AIPrivateInterfaceServiceClient(server_address, cert_file))
        self.log.info("AIProcessingService initialized successfully.")

    def Process(self, request_iterator, context):
        """Handles incoming requests and dispatches them based on payload type."""
        authorization_token, user_locale = MetadataUtils.extract(context)

        if not MetadataUtils.validate(authorization_token, user_locale, context):
            self.log.error("Metadata Invalidated")
            return

        for request in request_iterator:
            try:
                payload_type = request.WhichOneof("payload")
                if payload_type == "initialize":
                    yield from self.request_handler.handle_initialize_request(request)
                elif payload_type == "audio_chunk":
                    yield from self.request_handler.handle_audio_chunk_request(request)
                elif payload_type == "finalize":
                    yield from self.request_handler.handle_finalize_request(request, authorization_token)
                else:
                    raise ValueError("Invalid Request Type")
            except ValueError as e:
                self.log.error(str(e))
                yield ErrorResponse.generate_invalid_request_response()
            except Exception as e:
                self.log.error(f"Unexpected error in Process method: {str(e)}")
                yield ErrorResponse.generate_internal_server_error_response()

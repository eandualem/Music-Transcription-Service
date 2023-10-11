import grpc
from logger import Logger
from Declarations.Service import AIProcessingService_pb2_grpc


class AIProcessingServiceClient:
    """
    A client for the AI Processing Service.
    """

    def __init__(self, server_address, cert_file):
        # Load the server's certificate
        with open(cert_file, "rb") as f:
            trusted_certs = f.read()

        self.server_address = server_address
        self.log = Logger.get_logger(__name__)

        # Create gRPC channel credentials using the server's certificate
        credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)

        # Create a secure channel for communication
        options = [("grpc.max_receive_message_length", 20 * 1024 * 1024)]  # 20 MB
        self.channel = grpc.secure_channel(server_address, credentials, options=options)
        self.stub = AIProcessingService_pb2_grpc.AIProcessingServiceStub(self.channel)

    def process(self, requests, authorization_token, user_locale, timeout=None):
        metadata = [("authorization", authorization_token), ("user-locale", user_locale)]

        try:
            response_iterator = self.stub.Process(iter(requests), metadata=metadata, timeout=timeout)
            for response in response_iterator:
                yield response
        except grpc.RpcError as e:
            # Handle specific error codes for better feedback
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                self.log.error("Authentication failed.")
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                self.log.error("Server is unavailable.")
            else:
                self.log.error(f"Error during processing with server {self.server_address}: {e.details()}")
            raise RuntimeError(
                f"Error communicating with the server {self.server_address}. Details: {e.details()}"
            ) from e

    def close(self):
        """
        Close the gRPC channel.
        """
        if self.channel:
            self.channel.close()
            self.log.info(f"Closed gRPC channel with server {self.server_address}.")

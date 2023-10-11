import grpc
from config import Config
from logger import Logger
from concurrent import futures
from ai_processing_service import AIProcessingService
from Declarations.Service import AIProcessingService_pb2_grpc


class Server:
    def __init__(self, config):
        self.config = config
        self.log = Logger.get_logger(__name__)
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.GRPC_SERVER_MAX_WORKERS),
            options=[("grpc.max_send_message_length", self.config.GRPC_MAX_SEND_MESSAGE_LENGTH)],
        )

    def _initialize_service(self):
        AIProcessingService_pb2_grpc.add_AIProcessingServiceServicer_to_server(
            AIProcessingService(
                server_address=self.config.AI_PRIVATE_INTERFACE_SERVER_ADDRESS,
                cert_file=self.config.AI_PRIVATE_INTERFACE_CERT_FILE,
            ),
            self.server,
        )

    def _set_server_credentials(self):
        # Load the server's certificate and private key
        with open(self.config.GRPC_SERVER_CERT_FILE, "rb") as f:
            server_cert = f.read()
        with open(self.config.GRPC_SERVER_KEY_FILE, "rb") as f:
            server_key = f.read()

        # Create server credentials
        server_credentials = grpc.ssl_server_credentials([(server_key, server_cert)])
        self.server.add_secure_port(f"[::]:{self.config.GRPC_SERVER_PORT}", server_credentials)

    def start(self):
        try:
            self._initialize_service()
            if self.config.GRPC_SERVER_CERT_FILE and self.config.GRPC_SERVER_KEY_FILE:
                self._set_server_credentials()
            else:
                self.log.error(
                    "Starting server in insecure mode is not allowed. Please provide the necessary certificates."
                )
                return
            self.server.start()
            self.log.info(f"AIProcessingService server started and listening on port {self.config.GRPC_SERVER_PORT}.")
            self._await_termination()
        except grpc.RpcError as rpc_err:
            self.log.error(f"gRPC error occurred: {rpc_err}")
        except ValueError as val_err:
            self.log.error(f"Value error occurred: {val_err}")
        except Exception as e:
            self.log.error(f"An unexpected error occurred while starting the AIProcessingService server: {e}")

    def _await_termination(self):
        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            self.log.info("Attempting graceful shutdown...")
            self.server.stop(10)  # 10 seconds grace period for shutdown
            self.log.info("AIProcessingService server stopped.")


if __name__ == "__main__":
    config = Config()
    server = Server(config=config)
    server.start()

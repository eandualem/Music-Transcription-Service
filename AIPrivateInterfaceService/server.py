import grpc
from logger import Logger
from concurrent import futures
from ai_private_interface_service import AIPrivateInterfaceService
from Declarations.Service import AIPrivateInterfaceService_pb2_grpc


def serve(cert_file=None, key_file=None):
    # Create a gRPC server with a specific number of worker threads
    log = Logger.get_logger("AIPrivateInterfaceService")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add the services to the server
    AIPrivateInterfaceService_pb2_grpc.add_AIPrivateInterfaceServiceServicer_to_server(
        AIPrivateInterfaceService(), server
    )

    # Read in key and certificate for secure server connection
    with open(key_file, "rb") as f:
        private_key = f.read()
    with open(cert_file, "rb") as f:
        certificate_chain = f.read()

    # Create server credentials
    server_credentials = grpc.ssl_server_credentials(
        (
            (
                private_key,
                certificate_chain,
            ),
        )
    )

    # Start the server on a specific port with the SSL credentials
    server.add_secure_port("[::]:50057", server_credentials)
    server.start()
    log.info("AIPrivateInterfaceService server started and listening on port 50057.")
    server.wait_for_termination()


if __name__ == "__main__":
    serve(cert_file="certificate.pem", key_file="private.pem")

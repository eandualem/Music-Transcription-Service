import grpc
from logger import Logger
from google.protobuf.any_pb2 import Any
from Declarations.Model.Report_pb2 import Report
from Declarations.Model.KeyValue_pb2 import KeyValue
from Declarations.Model.OpaqueContainer_pb2 import OpaqueContainer
from Declarations.Service import AIPrivateInterfaceService_pb2_grpc
from Declarations.Model.AIPrivateInterfaceService.ReportResult_pb2 import ReportResultRequest
from Declarations.Model.AIPrivateInterfaceService.FetchInitialData_pb2 import FetchInitialDataRequest


class AIPrivateInterfaceServiceClient:
    """
    A gRPC client for the AIPrivateInterfaceService, providing methods to fetch initial data,
    report processing results, and manage the underlying gRPC channel.
    """

    def __init__(self, server_address: str, cert_file: str, always_use_secure_channel: bool = True):
        self.log = Logger.get_logger(__name__)
        self._initialize_grpc_channel(server_address, cert_file, always_use_secure_channel)

    def _initialize_grpc_channel(self, server_address: str, cert_file: str, always_use_secure_channel: bool):
        """Initialize the gRPC channel based on provided arguments."""
        if always_use_secure_channel and not cert_file:
            raise ValueError("Secure channel requested, but no certificate file provided.")

        credentials = self._get_channel_credentials(cert_file)
        self.channel = (
            grpc.secure_channel(server_address, credentials) if credentials else grpc.insecure_channel(server_address)
        )
        self.stub = AIPrivateInterfaceService_pb2_grpc.AIPrivateInterfaceServiceStub(self.channel)

    def _get_channel_credentials(self, cert_file: str) -> grpc.ChannelCredentials:
        """Retrieve gRPC channel credentials from a certificate file."""
        try:
            with open(cert_file, "rb") as cert:
                return grpc.ssl_channel_credentials(cert.read())
        except Exception as e:
            self.log.error(f"Error reading the certificate file {cert_file}: {str(e)}")
            raise

    def _pack_into_opaque_container(self, client_token: str) -> OpaqueContainer:
        """Pack a client token into an OpaqueContainer for gRPC communication."""
        kv = KeyValue(key="client_token", value=client_token)
        any_message = Any()
        any_message.Pack(kv)
        container = OpaqueContainer()
        container.opaque.CopyFrom(any_message)
        return container

    def fetch_initial_data(self, client_token: str) -> FetchInitialDataRequest:
        """Fetch initial data using a client token."""
        try:
            token_container = self._pack_into_opaque_container(client_token)
            request = FetchInitialDataRequest(client_token=token_container)
            response = self.stub.FetchInitialData(request)
            # self.log.info(f"Fetched initial data with response: {response}.")
            return response
        except grpc.RpcError as e:
            self.log.error(f"Error fetching initial data: {e.details()}")
            raise
        except Exception as e:
            self.log.error(f"Unexpected error while fetching initial data: {str(e)}")
            raise

    def report_processing_result(self, report: Report, client_token: str) -> ReportResultRequest:
        """Report processing results using a client token and report data."""
        try:
            token_container = self._pack_into_opaque_container(client_token)
            request = ReportResultRequest(report=report, client_token=token_container)
            response = self.stub.ReportResult(request)
            # self.log.info(f"Reported result with response: {response}.")
            return response
        except grpc.RpcError as e:
            self.log.error(f"Error reporting result: {e.details()}")
            raise
        except Exception as e:
            self.log.error(f"Unexpected error while reporting result: {str(e)}")
            raise

    def close(self):
        """Close the gRPC channel."""
        if self.channel and self.channel._channel.check_connectivity_state(True) != grpc.ChannelConnectivity.SHUTDOWN:
            self.channel.close()
            self.log.info("Closed gRPC channel.")

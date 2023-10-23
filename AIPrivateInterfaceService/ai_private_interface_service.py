import grpc
from logger import Logger
from Declarations.Service import AIPrivateInterfaceService_pb2_grpc
from Declarations.Model.AIPrivateInterfaceService import FetchInitialData_pb2, ReportResult_pb2


class AIPrivateInterfaceService(AIPrivateInterfaceService_pb2_grpc.AIPrivateInterfaceServiceServicer):
    logger = Logger.get_logger("AIPrivateInterfaceService")
    LYRICS_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1bJqfK7AAIkK9mOewg6f605VlKbdgHQ8B"
    TRACK_DOWNLOAD_URL = " https://drive.google.com/uc?export=download&id=1nUVv_ogF_q-lV14PFrY7wTg5rc3Aorlu"
    VOICE_HELPER_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1jjSa1gMteNvNqqZGHOTc9WB_QZfr6PkD"

    def _handle_exception(self, e, context):
        """Centralized exception handling."""
        self.logger.error(f"Exception in AIPrivateInterfaceService: {e}")
        context.set_details(str(e))
        context.set_code(grpc.StatusCode.INTERNAL)
        raise e  # Re-raise the exception to let gRPC handle it

    def FetchInitialData(self, request, context):
        """Fetch initial data like lyrics, track, and voice helper URLs."""
        try:
            self.logger.info(f"Received FetchInitialData request with client token: {request.client_token}")
            return FetchInitialData_pb2.FetchInitialDataResponse(
                lyrics_download_url=self.LYRICS_DOWNLOAD_URL,
                track_download_url=self.TRACK_DOWNLOAD_URL,
                voice_helper_download_url=self.VOICE_HELPER_DOWNLOAD_URL,
            )
        except Exception as e:
            self._handle_exception(e, context)

    def ReportResult(self, request, context):
        """Handle report results from the client."""
        try:
            # Check if the report contains actual report data
            if request.report:
                self.logger.info(f"Received report data: {request.report}")
            # Check if the report contains error data
            elif request.errors:
                self.logger.info(f"Received error data: {request.errors}")  # Corrected the logging statement
            else:
                self.logger.warning("Received report with neither report data nor error data.")
                context.set_details("Invalid report received.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return ReportResult_pb2.ReportResultResponse()

            return ReportResult_pb2.ReportResultResponse()
        except Exception as e:
            self._handle_exception(e, context)

import grpc
from config import Config
from Declarations.Model.AIProcessingService import AIProcessingResponse_pb2


class ErrorResponse:
    @staticmethod
    def generate_error_response(
        code: grpc.StatusCode,
        details: str,
    ):
        """Generates and returns an AI processing response for error scenarios.

        Args:
            code (grpc.StatusCode): The gRPC status code for the error.
            details (str): The error details message.

        Returns:
            AIProcessingResponse: The constructed error response.
        """
        return AIProcessingResponse_pb2.AIProcessingResponse(
            status_code=code,
            error=AIProcessingResponse_pb2.AIProcessingResponse.Error(
                correlation_id=Config.DEFAULT_CORRELATION_ID,
                debug_description="Invalid request received by server.",
                user_description=details,
            ),
        )

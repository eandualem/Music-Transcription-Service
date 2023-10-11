from google.protobuf.any_pb2 import Any
from Declarations.Model.KeyValue_pb2 import KeyValue
from Declarations.Model.OpaqueContainer_pb2 import OpaqueContainer
from Declarations.Model.AIProcessingService.AIProcessingRequest_pb2 import AIProcessingRequest


class RequestBuilder:
    """Class to build different types of AIProcessing requests."""

    @staticmethod
    def create_initialize_request(client_token: str) -> AIProcessingRequest:
        """Construct an initialization request with a packed KeyValue message."""
        kv = KeyValue(key="client_token", value=client_token)
        any_message = Any()
        any_message.Pack(kv)
        container = OpaqueContainer()
        container.opaque.CopyFrom(any_message)
        request = AIProcessingRequest()
        request.initialize.client_token.CopyFrom(container)
        return request

    @staticmethod
    def create_audio_chunk_request(raw_audio_data: bytes) -> AIProcessingRequest:
        """Construct a request containing an audio chunk."""
        request = AIProcessingRequest()
        request.audio_chunk.audio_data = raw_audio_data
        return request

    @staticmethod
    def create_finalize_request(reason: int) -> AIProcessingRequest:
        """Construct a finalization request with a given reason."""
        request = AIProcessingRequest()
        request.finalize.finalize_reason = reason
        return request

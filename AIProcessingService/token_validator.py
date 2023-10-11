from Declarations.Model.KeyValue_pb2 import KeyValue
from Declarations.Model.OpaqueContainer_pb2 import OpaqueContainer


class TokenValidator:
    @staticmethod
    def unpack_client_token(container: OpaqueContainer) -> str:
        """Unpacks and returns the client token from the provided OpaqueContainer.

        Args:
            container (OpaqueContainer): The container holding the opaque token data.

        Returns:
            str: The unpacked client token value.
        """

        try:
            any_message = container.opaque
            kv = KeyValue()
            any_message.Unpack(kv)
            return kv.value
        except Exception as e:
            raise ValueError(f"Error unpacking client token: {str(e)}")

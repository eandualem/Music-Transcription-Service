import grpc
from config import Config
from logger import Logger


class MetadataUtils:
    log = Logger.get_logger(__name__)
    config = Config()

    @staticmethod
    def extract(context):
        """Extracts authorization and user-locale metadata from the given context.

        Args:
            context: The gRPC context from which to extract metadata.

        Returns:
            tuple: A tuple containing the authorization token and user locale.
        """

        metadata = dict(context.invocation_metadata())
        return metadata.get(MetadataUtils.config.AUTHORIZATION_KEY), metadata.get(MetadataUtils.config.USER_LOCALE_KEY)

    @staticmethod
    def validate(authorization_token, user_locale, context):
        """Validates the presence of the authorization token and user locale in the metadata.

        Args:
            authorization_token (str): The extracted authorization token.
            user_locale (str): The extracted user locale.
            context: The gRPC context for setting error codes and details.

        Returns:
            bool: True if both values are present, False otherwise.
        """

        if not authorization_token:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details(MetadataUtils.config.MISSING_AUTHORIZATION_MSG)
            return False

        # TODO: Additional authorization token validation

        if not user_locale:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(MetadataUtils.config.MISSING_USER_LOCALE_MSG)
            return False

        return True

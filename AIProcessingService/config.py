import os
from dotenv import load_dotenv
import threading


class Config:
    """
    A centralized configuration management class that loads and provides access to
    environment variables and other configuration values.
    """

    _instance = None
    _lock = threading.Lock()  # Thread lock for thread-safe Singleton initialization

    def __new__(cls):
        with cls._lock:  # Ensuring thread safety
            if cls._instance is None:
                cls._instance = super(Config, cls).__new__(cls)
                cls._instance.init_config()
        return cls._instance

    def init_config(self):
        load_dotenv()
        self.GRPC_SERVER_PORT = int(self._get_env_variable("GRPC_SERVER_PORT"))
        self.GRPC_SERVER_CERT_FILE = self._get_env_variable("GRPC_SERVER_CERT_FILE")
        self.GRPC_SERVER_KEY_FILE = self._get_env_variable("GRPC_SERVER_KEY_FILE")
        self.AI_PRIVATE_INTERFACE_SERVER_ADDRESS = self._get_env_variable("AI_PRIVATE_INTERFACE_SERVER_ADDRESS")
        self.AI_PRIVATE_INTERFACE_CERT_FILE = self._get_env_variable("AI_PRIVATE_INTERFACE_CERT_FILE")
        self.GRPC_SERVER_MAX_WORKERS = int(self._get_env_variable("GRPC_SERVER_MAX_WORKERS"))
        self.GRPC_MAX_SEND_MESSAGE_LENGTH = int(self._get_env_variable("GRPC_MAX_SEND_MESSAGE_LENGTH"))
        self.DEFAULT_CORRELATION_ID = self._get_env_variable("DEFAULT_CORRELATION_ID")
        self.AUTHORIZATION_KEY = self._get_env_variable("AUTHORIZATION_KEY")
        self.USER_LOCALE_KEY = self._get_env_variable("USER_LOCALE_KEY")
        self.MISSING_AUTHORIZATION_MSG = (self._get_env_variable("MISSING_AUTHORIZATION_MSG"),)
        self.MISSING_USER_LOCALE_MSG = self._get_env_variable("MISSING_USER_LOCALE_MSG")

    @staticmethod
    def _get_env_variable(var_name):
        """Retrieve environment variable or raise error if not found."""
        value = os.environ.get(var_name)
        if not value:
            raise ValueError(f"Missing essential configuration: {var_name}. Please set this environment variable.")
        return value

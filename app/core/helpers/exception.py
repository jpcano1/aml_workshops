from http import HTTPStatus


class ControllerException(Exception):
    def __init__(self, http_code: int, message: str) -> None:
        """
        Initializer for Controller custom Exception.

        :param http_code: The HTTP status code of the error
        :param message: The message of the error
        """
        super().__init__(message)
        self.http_code = http_code
        self.message = message

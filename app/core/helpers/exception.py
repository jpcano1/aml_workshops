from http import HTTPStatus


class ControllerException(Exception):
    def __init__(self, http_code: HTTPStatus, message: str) -> None:
        """
        Initializer for Controller custom Exception.

        :param http_code: The HTTP status code of the error
        :param message: The message of the error
        """
        self.http_code = http_code
        self.message = message
        super(message)

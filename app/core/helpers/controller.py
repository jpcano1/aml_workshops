from typing import Any


class BaseController:
    _kwargs: dict[str, Any]
    response: dict[str, Any]

    def __init__(self, **kwargs: dict[str, Any]):
        self._kwargs = kwargs
        self.response = {}
        self.__parse_kwargs()

    def __parse_kwargs(self):
        for key, value in self._kwargs.items():
            setattr(self, key, value)

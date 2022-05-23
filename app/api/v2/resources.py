from http import HTTPStatus

from flask_restful import Resource, abort
from webargs import fields, validate
from webargs.flaskparser import use_kwargs


class Health(Resource):
    def get(self, *args) -> tuple[str, int]:
        """
        Health check route.

        ---
        tags:
            - v2
        responses:
            200:
                description: Hello world!
        """
        return "Hello world", HTTPStatus.OK

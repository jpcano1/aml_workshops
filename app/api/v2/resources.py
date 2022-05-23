from http import HTTPStatus
from typing import Any

from flask_restful import Resource, abort
from sklearn.svm import SVC
from webargs import fields, validate
from webargs.flaskparser import use_kwargs

from app.core.helpers.exception import ControllerException

from ..controllers import EntregaFinalController


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


class Prediction(Resource):
    def __init__(self, model: SVC) -> None:
        self.model = model

    @use_kwargs(
        {"image": fields.Raw(required=True)},
        location="files",
    )
    def post(self, **kwargs) -> tuple[dict[str, Any], int]:
        """
        Predict images.

        ---
        tags:
          - v2
        consumes:
          - multipart/form-data
        parameters:
          - in: formData
            name: image
            type: file
            description: The file to upload
        responses:
            200:
                description: Here's your prediction
        """
        controller = EntregaFinalController(model=self.model, **kwargs)

        try:
            controller.handle_post()
        except ControllerException as err:
            abort(err.http_code, error=err.message)
        return controller.response, HTTPStatus.OK

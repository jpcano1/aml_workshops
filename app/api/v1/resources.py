from http import HTTPStatus

from flask_restful import Resource, abort
from webargs import fields, validate
from webargs.flaskparser import use_kwargs

from app.core.helpers.exception import ControllerException

from ..controllers import Taller1Controller


class Health(Resource):
    def get(self):
        """
        Health check route.

        ---
        tags:
            - v1
        responses:
            200:
                description: Hello world!
        """
        return "Hello world", HTTPStatus.OK


class Taller1(Resource):
    @use_kwargs(
        {
            "geo_lat": fields.Float(required=True, data_key="latitude"),
            "geo_lon": fields.Float(required=True, data_key="longitude"),
            "region": fields.Int(required=True),
            "building_type": fields.Int(required=True, validate=lambda x: 0 <= x <= 5),
            "level": fields.Int(required=True, validate=lambda x: x > 0),
            "levels": fields.Int(required=True, validate=lambda x: x > 0),
            "rooms": fields.Int(required=True, validate=lambda x: x >= -1),
            "area": fields.Int(required=True, validate=lambda x: x >= 5),
            "kitchen_area": fields.Int(required=True, validate=lambda x: x >= 1),
            "object_type": fields.Int(required=True, validate=validate.OneOf([1, 2])),
        },
        location="json",
    )
    def post(self, **kwargs):
        """
        Taller 1 prediction endpoint.

        ---
        tags:
            - v1
        parameters:
          - in: body
            name: body
            required: true
            schema:
                type: object
                properties:
                    latitude:
                        type: number
                        required: true
                    longitude:
                        type: number
                        required: true
                    region:
                        type: number
                        required: true
                    building_type:
                        type: number
                        required: true
                    level:
                        type: number
                        required: true
                    levels:
                        type: number
                        required: true
                    rooms:
                        type: number
                        required: true
                    area:
                        type: number
                        required: true
                    kitchen_area:
                        type: number
                        required: true
                    object_type:
                        type: number
                        required: true
        responses:
            200:
                description: Prediction completed
        """
        controller = Taller1Controller(**kwargs)

        try:
            controller.handle_post()
        except ControllerException as err:
            abort(err.http_code, error=err.message)
        return controller.response, HTTPStatus.OK

from http import HTTPStatus

from flask_restful import Resource, abort
from webargs import fields, validate
from webargs.flaskparser import use_kwargs


class Health(Resource):
    def get(self):
        """
        Health check route.

        ---
        tags:
            - Health check
        responses:
            200:
                description: Hello world!
        """
        return "Hello world", HTTPStatus.OK


class Taller1(Resource):
    @use_kwargs(
        {
            "geo_lat": fields.Float(required=True, data_key="longitude"),
            "geo_lon": fields.Float(required=True, data_key="latitude"),
            "region": fields.Int(required=True),
            "building_type": fields.Int(required=True, validate=lambda x: 0 <= x <= 5),
            "level": fields.Int(required=True, validate=lambda x: x > 0),
            "levels": fields.Int(required=True, validate=lambda x: x > 0),
            "rooms": fields.Int(required=True, validate=lambda x: x >= -1),
            "area": fields.Int(required=True, validate=lambda x: x >= 5),
            "kitchen_area": fields.Int(required=True, validate=lambda x: x >= 1),
            "object_type": fields.Int(required=True, validate=validate.OneOf([1, 2])),
        }
    )
    def post(self, **kwargs):
        ...

import resources
from flask import Blueprint
from flask_restful import Api

v1 = Blueprint("v1", __name__)
api = Api(v1, catch_all_404s=True)

"""
Taller 1
"""
api.add_resource(resources.Taller1, "/taller_1", endpoint="taller_1")

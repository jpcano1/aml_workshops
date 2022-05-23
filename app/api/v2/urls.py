from flask import Blueprint
from flask_restful import Api

from app.core.helpers.model_loader import model_loader

from . import resources

v2 = Blueprint("v2", __name__)
api = Api(v2, catch_all_404s=True)


model = model_loader()
"""
Health
"""
api.add_resource(resources.Health, "/", endpoint="healthcheck")

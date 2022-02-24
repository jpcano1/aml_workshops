from flasgger import Swagger
from flask import Flask
from flask_cors import CORS

from ..config import Development, Production, Testing
from ..swagger import swagger_config, swagger_template


def init_app() -> Flask:
    """Initialize of base app."""
    flask_app = Flask(__name__, instance_relative_config=True)

    if flask_app.env == "production":
        flask_app.config.from_object(Production)
    elif flask_app.env == "testing":
        flask_app.config.from_object(Testing)
    else:
        flask_app.config.from_object(Development)

    Swagger(
        flask_app,
        template=swagger_template({}),
        config=swagger_config(),
    )

    CORS(flask_app)

    return flask_app

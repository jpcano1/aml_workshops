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

    CORS(flask_app)

    Swagger(
        flask_app,
        template=swagger_template({}),
        config=swagger_config(),
    )

    from .v1 import v1
    from .v2 import v2

    flask_app.register_blueprint(v1, url_prefix="/v1")
    flask_app.register_blueprint(v2, url_prefix="/v2")

    return flask_app

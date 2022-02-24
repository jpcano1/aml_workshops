from typing import Any


def swagger_template(
    definitions: dict[str, Any],
    base_path: str = "/",
) -> dict[str, Any]:
    """
    Describe the base configuration template.

    :param definitions: Swagger template definitions
    :param base_path: The base path for routes
    :return: The swagger template
    """
    return {
        "swagger": "2.0",
        "consumes": [
            "application/json",
        ],
        "produces": [
            "application/json",
        ],
        "definitions": definitions,
        "schemes": ["https", "http"],
        "base_path": base_path,
        "securityDefinitions": {
            "Bearer": {
                "description": "JWT Bearer token",
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
            },
        },
        "security": [{"Bearer": []}],
    }


def swagger_config(
    base_path: str = "/",
) -> dict[str, Any]:
    """
    Describe the base config.

    :param base_path: Base path for routes
    :return: The swagger config
    """
    return {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": f"{base_path}/apispec.json",
                "rule_filter": lambda _: True,
                "model_filter": lambda _: True,
            }
        ],
        "static_url_path": f"{base_path}/flasgger_static",
        "swagger_ui": True,
        "specs_route": f"{base_path}/apidocs",
    }

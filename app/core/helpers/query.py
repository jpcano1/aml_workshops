import json
import os
from typing import Any

import boto3


def query(input_json: str, app_name: str) -> Any:
    """
    Function to make predictions to Sagemaker.

    :param input_json: Pandas-Split formatted JSON to pass to the endpoint
    :param app_name: The name of the app endpoint
    """
    client = boto3.client(
        "sagemaker-runtime",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=input_json,
        ContentType="application/json; format=pandas-split",
    )

    return json.loads(response["Body"].read().decode("ascii"))

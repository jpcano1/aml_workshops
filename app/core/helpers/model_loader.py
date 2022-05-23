import os

import gdown
import joblib
from loguru import logger
from sklearn.svm import SVC


def model_loader() -> SVC:
    """Load model from Drive."""
    logger.info("Loading model...")

    output_file = "best_model.pkl"
    model_id = os.getenv("MODEL_ID")

    if not model_id:
        raise RuntimeError("Model ID is not proportioned")

    gdown.download(id=model_id, output=output_file, quiet=False)
    model: SVC = joblib.load(output_file)

    try:
        logger.info("Deleting model...")
        os.remove(output_file)
    except FileNotFoundError:
        raise RuntimeError("File not found")

    return model

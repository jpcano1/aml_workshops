import mimetypes
from http import HTTPStatus

import cv2
import numpy as np
from skimage import color
from sklearn.svm import SVC
from werkzeug.datastructures import FileStorage

from app.core.helpers.controller import BaseController
from app.core.helpers.exception import ControllerException
from app.core.helpers.features import LBP


class EntregaFinalController(BaseController):

    image: FileStorage
    model: SVC
    class_dict = {
        0: "COVID",
        1: "Lung Opacity",
        2: "Normal",
        3: "Viral Pneumonia",
    }

    def handle_post(self):
        if self.image.content_type not in [
            mimetypes.types_map[".png"],
            mimetypes.types_map[".jpg"],
            mimetypes.types_map[".jpeg"],
        ]:
            raise ControllerException(HTTPStatus.BAD_REQUEST, "Not a valid image type")

        image_string = self.image.stream.read()
        np_img = np.fromstring(image_string, np.uint8)  # type: ignore
        img = cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)

        if len(img.shape) > 2:
            img = color.rgb2gray(img)

        img_resized = cv2.resize(img, (150, 150), cv2.INTER_CUBIC)
        hog = LBP()
        img_features = hog.process_single_image(img_resized)

        prediction = self.model.predict([img_features])

        self.response = {"response": f"Prediction was {self.class_dict[prediction[0]]}"}

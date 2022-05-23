from typing import Union

import numpy as np
from skimage import feature, morphology


class HoG:
    def __init__(self) -> None:
        self.pixels_per_cell = (4, 4)
        self.orientations = 4

    def process_single_image(self, img: np.ndarray) -> np.ndarray:
        """
        Process single image to get the HoG.

        :param img: The image to be processed
        :return: The HoG of the image processed
        """
        img = morphology.opening(img, selem=morphology.disk(3))
        return feature.hog(
            img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            multichannel=False,
        )

    def process_multiple_images(
        self, images: Union[list[np.ndarray], np.ndarray]
    ) -> list[np.ndarray]:
        """
        Process images per batch.

        :param images: The images to be processed
        :return: The HoG of the batch of images
        """
        transformed_images = []
        for img in images:
            img = self.process_single_image(img)
            transformed_images.append(img)
        return transformed_images

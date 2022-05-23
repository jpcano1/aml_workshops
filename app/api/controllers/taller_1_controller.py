import os

import pandas as pd

from app.core.helpers.controller import BaseController
from app.core.helpers.query import query


class Taller1Controller(BaseController):
    geo_lat: float
    geo_lon: float
    region: int
    building_type: int
    level: int
    levels: int
    rooms: int
    area: int
    kitchen_area: int
    object_type: int

    def handle_post(self):
        data = pd.DataFrame(
            {
                "geo_lat": self.geo_lat,
                "geo_lon": self.geo_lon,
                "region": self.region,
                "building_type": self.building_type,
                "level": self.level,
                "levels": self.levels,
                "rooms": self.rooms,
                "area": self.area,
                "kitchen_area": self.kitchen_area,
                "object_type": self.object_type,
            },
            index=[1],
        )

        app_name = os.getenv("AWS_SAGEMAKER_APP_NAME")

        result = query(data.to_json(orient="split"), app_name)
        self.response = {
            "response": f"The price is: {result[0]:.5f}",
        }

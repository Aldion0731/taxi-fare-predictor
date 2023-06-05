from enum import Enum


class ScalingConstants(float, Enum):
    NOMINAL_MIN_LATITUDE = 37.0
    NOMINAL_MIN_LONGITUDE = -78.0
    NOMINAL_RANGE = 8.0
    # Motivated by N.Y.C latitude and longitude ranging between (37, 45) and (-70, -78) respectively


class LatLonColumns(list, Enum):
    LON_COLS = ["pickup_longitude", "dropoff_longitude"]
    LAT_COLS = ["pickup_latitude", "dropoff_latitude"]

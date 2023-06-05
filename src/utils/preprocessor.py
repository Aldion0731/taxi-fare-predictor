import tensorflow as tf
from keras import layers

from .preprocessing_constants import LatLonColumns, ScalingConstants


def transform_features(inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    transformed = inputs.copy()

    for lon_col in LatLonColumns.LON_COLS:
        transformed[lon_col] = layers.Lambda(scale_longitude, name=f"scaled_{lon_col}")(
            inputs[lon_col]
        )

    for lat_col in LatLonColumns.LAT_COLS:
        transformed[lat_col] = layers.Lambda(scale_latitude, name=f"scaled_{lat_col}")(
            inputs[lat_col]
        )

    distance_measure_inputs = {
        k: v
        for k, v in inputs.items()
        if k in LatLonColumns.LAT_COLS + LatLonColumns.LON_COLS
    }

    transformed["euclidean_distance"] = layers.Lambda(
        compute_euclidean_distance, name="euclidean_distance"
    )(distance_measure_inputs)

    return transformed


def scale_longitude(lon_column: tf.Tensor) -> tf.Tensor:
    return (
        lon_column - ScalingConstants.NOMINAL_MIN_LONGITUDE
    ) / ScalingConstants.NOMINAL_RANGE


def scale_latitude(lat_column: tf.Tensor) -> tf.Tensor:
    return (
        lat_column - ScalingConstants.NOMINAL_MIN_LATITUDE
    ) / ScalingConstants.NOMINAL_RANGE


def compute_euclidean_distance(distance_params: dict[str, tf.Tensor]) -> tf.Tensor:
    lon_diff = (
        distance_params["dropoff_longitude"] - distance_params["pickup_longitude"]
    )
    lat_diff = distance_params["dropoff_latitude"] - distance_params["pickup_latitude"]
    return tf.sqrt(tf.pow(lon_diff, 2) + tf.pow(lat_diff, 2))

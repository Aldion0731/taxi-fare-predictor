from dataclasses import asdict, dataclass

from keras import layers
from keras.engine.keras_tensor import KerasTensor
from keras.metrics import RootMeanSquaredError
from keras.models import Model

from .preprocessor import transform_features

FEATURE_COLS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "hourofday",
    "dayofweek",
]


@dataclass
class CompileArgs:
    optimizer: str
    loss: str
    metrics: list


COMPILE_ARGS = CompileArgs(
    optimizer="adam", loss="mse", metrics=[RootMeanSquaredError(name="rmse")]
)


def build_baseline_model(feature_cols: list[str] = FEATURE_COLS) -> Model:
    inputs = build_input_layer(feature_cols)
    output = build_dense_layers(inputs)

    model = Model(inputs, output)
    model.compile(**asdict(COMPILE_ARGS))
    return model


def build_preprocessing_layer_model(feature_cols: list[str] = FEATURE_COLS) -> Model:
    inputs = build_input_layer(feature_cols)
    transformed = transform_features(inputs)
    output = build_dense_layers(transformed)

    model = Model(inputs, output)
    model.compile(**asdict(COMPILE_ARGS))
    return model


def build_input_layer(feature_cols: list[str]) -> dict[str, KerasTensor]:
    return {
        col_name: layers.Input(name=col_name, shape=(1), dtype="float32")
        for col_name in feature_cols
    }


def build_dense_layers(inputs: dict[str, KerasTensor]) -> KerasTensor:
    concatenated_inputs = layers.Concatenate()(inputs.values())

    h1 = layers.Dense(32, activation="relu", name="h1")(concatenated_inputs)
    h2 = layers.Dense(8, activation="relu", name="h2")(h1)

    return layers.Dense(1, activation="linear", name="fare")(h2)

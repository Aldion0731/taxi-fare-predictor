from enum import Enum
from typing import Any

import tensorflow as tf

LABEL_COLUMN = "fare_amount"


class NNMode(Enum):
    EVAL = "eval"
    TRAIN = "train"


def load_dataset(file_pattern: str, batch_size: int, mode: NNMode = NNMode.EVAL) -> Any:
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern, batch_size=batch_size, label_name=LABEL_COLUMN
    )

    if mode is NNMode.TRAIN:
        dataset = dataset.shuffle(1000).repeat()
        dataset = dataset.prefetch(1)

    return dataset

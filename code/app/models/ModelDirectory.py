import os
import csv
from typing import Any, Dict, List, Optional
import tensorflow as tf


class ModelDirectory:
    """
    Utility functions for working with a model directory

    Model directory has the following structure:
    /
        checkpoints/
            checkpoint-0001/
            checkpoint-0002/
            checkpoint-{number of finished epochs}/
        visualizations/
            my-vis-0123.png
        metrics.csv
    """

    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    @property
    def checkpoints_path(self):
        return os.path.join(self.directory_path, "checkpoints")

    @property
    def checkpoint_format_path(self):
        return os.path.join(self.checkpoints_path, "checkpoint-{epoch:04d}")

    @property
    def visualizations_path(self):
        return os.path.join(self.directory_path, "visualizations")

    @property
    def metrics_csv_path(self):
        return os.path.join(self.directory_path, "metrics.csv")

    def assert_folder_structure(self):
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.visualizations_path, exist_ok=True)

    def load_latest_checkpoint(self, custom_objects: dict) -> Any:
        if not os.path.isdir(self.checkpoints_path):
            return None
        folders = os.listdir(self.checkpoints_path)
        prefix_len = len("checkpoint-")
        numbers = [int(f[prefix_len:]) for f in folders]
        if len(numbers) == 0:
            return None
        latest_epoch = max(numbers)
        model = tf.keras.models.load_model(
            os.path.join(
                self.checkpoints_path,
                "checkpoint-{:04d}".format(latest_epoch)
            ),
            custom_objects=custom_objects
        )
        return model

    def load_metrics_csv(self) -> List[Dict[str, float]]:
        try:
            with open(self.metrics_csv_path, "r") as f:
                return [
                    {k: float(v) for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)
                ]
        except IOError:
            return []

    def get_best_epoch_metrics_record(
        self,
        by_metric: str = "val_loss",
        search_for=min
    ) -> Optional[Dict[str, float]]:
        metrics = self.load_metrics_csv()
        if len(metrics) == 0:
            return None
        return search_for(metrics, key=lambda x: x[by_metric])

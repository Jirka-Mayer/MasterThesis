import argparse
import os


class Experiment:
    @property
    def name(self):
        return "unnamed-experiment"

    def describe(self):
        return None

    def define_arguments(self, parser: argparse.ArgumentParser):
        pass
    
    def run(self, args: argparse.Namespace):
        raise NotImplementedError("Experiment must override the `run` method.")

    def experiment_directory(self, subpath: str = "") -> str:
        """Returns path to an experiment subdir"""
        experiment_dir = os.path.join(
            "experiments-data", # in the pwd = "/code" folder of the repository
            self.name
        )
        return os.path.join(experiment_dir, subpath)

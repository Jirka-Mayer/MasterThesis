import argparse


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

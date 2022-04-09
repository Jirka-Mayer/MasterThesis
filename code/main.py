import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import argparse
from app.experiments.experiment_list import EXPERIMENT_LIST

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(
    dest="experiment_name",
    metavar="experiment_name"
)

for e in EXPERIMENT_LIST:
    subparser = subparsers.add_parser(e.name)
    subparser.description = e.describe()
    e.define_arguments(subparser)

args = parser.parse_args()

if args.experiment_name is None:
    print("Available experiment names are:")
    for e in EXPERIMENT_LIST:
        print("\t" + e.name)
    exit(0)

for e in EXPERIMENT_LIST:
    if e.name == args.experiment_name:
        e.run(args)
        break

import os

DATASETS_PATH = os.path.expanduser("~/Datasets")

MUSCIMAPP_ANNOTATIONS = os.path.join(
    DATASETS_PATH,
    "MuscimaPlusPlus/v2.0/data/annotations"
)
MUSCIMAPP_TESTSET_INDEP = os.path.join(
    DATASETS_PATH,
    "MuscimaPlusPlus/v2.0/specifications/testset-independent.txt"
)
CVCMUSCIMA_IDEAL = os.path.join(
    DATASETS_PATH,
    "CvcMuscima_StaffRemoval/CvcMuscima-Distortions/ideal"
)

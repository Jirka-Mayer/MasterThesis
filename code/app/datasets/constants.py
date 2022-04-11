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

DEEPSCORES_PATH = os.path.join(
    DATASETS_PATH,
    "DeepScoresV2/ds2_dense"
)
DEEPSCORES_TEST_ANNOTATIONS = os.path.join(
    DEEPSCORES_PATH,
    "deepscores_test.json"
)
DEEPSCORES_TRAIN_ANNOTATIONS = os.path.join(
    DEEPSCORES_PATH,
    "deepscores_train.json"
)

from .Ex_Datasets import Ex_Datasets
from .Ex01_SemisupUnet import Ex01_SemisupUnet
from .Ex_HyperparamSearch import Ex_HyperparamSearch
from .Ex_Unet import Ex_Unet


EXPERIMENT_LIST = [
    #Ex01_SemisupUnet(), # legacy
    #Ex_HyperparamSearch(), # legacy
    Ex_Unet(),
    Ex_Datasets()
]

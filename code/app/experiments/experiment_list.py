from .Ex01_SemisupUnet import Ex01_SemisupUnet
from .Ex_HyperparamSearch import Ex_HyperparamSearch
from .Ex_KnowledgeTransfer import Ex_KnowledgeTransfer


EXPERIMENT_LIST = [
    Ex01_SemisupUnet(),
    Ex_HyperparamSearch(),
    Ex_KnowledgeTransfer()
]

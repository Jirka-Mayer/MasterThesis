# import the dataset
import importlib.util
__spec = importlib.util.spec_from_file_location(
    "datasets.01-muscima",
    "../00-datasets/01-muscima.py"
)
__module = importlib.util.module_from_spec(__spec)
__spec.loader.exec_module(__module)

# expose functions (such that code completion works)
def foobar():
    __module.foobar()

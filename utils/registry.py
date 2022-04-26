import importlib
import os
import glob
from pathlib import Path


class Registry:
    mapping_dict = {
        'models': {},
        'datasets': {}
    }

    @classmethod
    def register_models(cls, model_name):
        """
        Register a model to the class Registery with key of model_name
        :param model_name: model_name to be registered
        :return: the func itself
        """

        def wrapper(func):
            cls.mapping_dict['models'][model_name] = func
            return func

        return wrapper

    @classmethod
    def get_model(cls, name):
        assert name in cls.mapping_dict['models'].keys(), 'The model name is not registered or supported'
        return cls.mapping_dict["models"].get(name, None)


def setup_imports():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.join(root_folder, "..")
    model_folder = os.path.join(root_folder, "models")
    pattern_models = os.path.join(model_folder, '*.py')
    model_files = glob.glob(pathname=pattern_models, recursive=True)
    for file in model_files:
        file_name = Path(file).name
        name = file_name[:file_name.find('.py')]
        importlib.import_module(name='models.{}'.format(name))


registry = Registry()

from importlib import import_module
from activations.pytorch import PytorchVideoWrapper
from data_inputs.pytorch_data_inputs import PyTorchVideoDataInput

def pytorch_model(function, data_input_kwargs=None):
    module = import_module(f'torchvision.models')
    model_ctr = getattr(module, function)
    #TODO: remove datainputs dependencies
    data_inputs = PyTorchVideoDataInput(**data_input_kwargs)
    wrapper = PytorchVideoWrapper(data_inputs, identifier=function, model=model_ctr(pretrained=True), preprocessing=None)
    return wrapper
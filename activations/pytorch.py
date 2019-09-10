import torch
from collections import OrderedDict
from utils.mdl_utils import mask_unused_gpus
from model_tools.activations.pytorch import PytorchWrapper
from activations.video_core import VideoActivationsExtractorHelper

SUBMODULE_SEPARATOR = '.'

class PytorchVideoWrapper(PytorchWrapper):

	def __init__(self, data_inputs, model, preprocessing, identifier=None, *args, **kwargs):
		super(PytorchVideoWrapper, self).__init__(model, preprocessing, identifier, *args, **kwargs)
		devices = mask_unused_gpus()
		print(devices)
		self._device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
		self._model = model
		self.data_parallel = False

		if torch.cuda.device_count() > 1:
			self.data_parallel = True
			self._model = torch.nn.DataParallel(self._model, device_ids=devices, output_device=devices[0])

		self._model.to(self._device)

		self._data_inputs = data_inputs
		#
		self._extractor = VideoActivationsExtractorHelper(data_inputs=self._data_inputs
		                                                  , batch_size=self._data_inputs.batch_size*self._data_inputs.fpv
		                                                  , identifier=self.identifier
		                                                  , get_activations=self.get_activations
		                                                  , preprocessing=preprocessing)
		self._extractor.insert_attrs(self)

	def get_layer(self, layer_name):
		if layer_name == 'logits':
			return self._output_layer()
		module = self._model
		if self.data_parallel:
			module = module.module
		for part in layer_name.split(SUBMODULE_SEPARATOR):
			module = module._modules.get(part)
			assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
		return module

	def _output_layer(self):
		module = self._model
		if self.data_parallel:
			module = module.module

		while module._modules:
			module = module._modules[next(reversed(module._modules))]
		return module

	def get_activations(self, images=None, layer_names=None):
		return self._get_activations(layer_names)

	def _get_activations(self, layer_names):
		data = next(self._data_inputs.iterator)
		images = data[0].to(self._device)
		stim_paths = list(data[1])
		self._model.eval()

		layer_results = OrderedDict()
		hooks = []

		for layer_name in layer_names:
			layer = self.get_layer(layer_name)
			hook = self.register_hook(layer, layer_name, target_dict=layer_results)
			hooks.append(hook)

		self._model(images)

		for hook in hooks:
			hook.remove()

		return stim_paths, layer_results
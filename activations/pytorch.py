from collections import OrderedDict
from activations.video_core import VideoActivationsExtractorHelper
from model_tools.activations.pytorch import PytorchWrapper

SUBMODULE_SEPARATOR = '.'

class PytorchVideoWrapper(PytorchWrapper):

	def __init__(self, data_inputs, *args, **kwargs):
		super(PytorchVideoWrapper, self).__init__(*args, **kwargs)

		# dataset = ...
		# self._data_inputs = dataloader(dataset)
		#TODO: remove datainputs dependencies
		self._data_inputs = data_inputs
		#
		self._extractor = VideoActivationsExtractorHelper(data_inputs=self._data_inputs
		                                                  , batch_size=self._data_inputs.batch_size*self._data_inputs.fpv
		                                                  , identifier=self.identifier
		                                                  , get_activations=self.get_activations
		                                                  , preprocessing=None)
		self._extractor.insert_attrs(self)

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
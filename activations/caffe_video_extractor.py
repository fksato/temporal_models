from .video_core import VideoActivationsExtractorHelper
from data_inputs.build_lmdb import VideoDBBuilder


class CaffeVideoWrapper:

	def __init__(self, model, identifier=None, *args, **kwargs):

		self._data_inputs = VideoDBBuilder(**kwargs)
		self.identifier = identifier
		self._model = model
		self._extractor = self._extractor = VideoActivationsExtractorHelper(data_inputs=self._data_inputs
		                                                  , identifier=self.identifier
		                                                  , get_activations=self.get_activations
		                                                  , *args, **kwargs)

	def get_activations(self, images=None, layer_names=None):
		pass

	def _get_activations(self, layer_names):
		pass
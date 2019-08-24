from collections import OrderedDict
from activations.video_core import VideoActivationsExtractorHelper
from model_tools.activations.tensorflow import TensorflowWrapper

# VIDEO ACTIVATION EXTRACTOR HELPER
class TensorflowVideowrapper(TensorflowWrapper):

	def __init__(self, data_inputs, *args, **kwargs):
		"""
		:param: identifier: model identifier
		:param inputs: model input
		:param endpoints: model nodes
		:param session: tf session
		:param layers: string of nodes from Res3D to get activations from
		"""
		super(TensorflowVideowrapper, self).__init__(*args, **kwargs)
		self._data_inputs = data_inputs # func partial

		self._extractor = VideoActivationsExtractorHelper(data_inputs=self._data_inputs
		                                                  , batch_size=self._data_inputs.batch_size
		                                                  , identifier=self.identifier
		                                                  , get_activations=self.get_activations
		                                                  , preprocessing=None)

		self._extractor.insert_attrs(self)


	def get_activations(self, images=None, layer_names=None):
		return self._get_activations(layer_names)

	def _get_activations(self, layer_names):
		data = self._session.run(self._data_inputs.next_elem)
		layer_tensors = OrderedDict((layer, self._endpoints[
			layer if (layer != 'logits' or layer in self._endpoints) else next(reversed(self._endpoints))])
									for layer in layer_names)
		layer_outputs = self._session.run(layer_tensors, feed_dict={self._inputs: data[0]})
		return data[1], layer_outputs

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
		self._data_inputs = data_inputs

		self._extractor = VideoActivationsExtractorHelper(data_inputs=self._data_inputs #, batch_size=batch_size
		                                                  , batch_size=self._data_inputs.batch_size
		                                                  , identifier=self.identifier
														  , get_activations=self.get_activations
														  , preprocessing=None)

		self._extractor.insert_attrs(self)

	"""
	def make_from_paths(self, paths):
		# init_op,
		self.iterator = video_data_inputs(paths, frames_per_video=self.fpv, block_starts=self.block_starts
		                                                , preprocess_type=self.preprocess_type
		                                                , labels=self.labels
		                                                , t_window=self._t_window
		                                                , im_height=self._height, im_width=self._width
		                                                , repeat=self.repeat, shuffle=self.shuffle
		                                                , batch_size=self.batch_size, prefetch=self.prefetch
		                                                , num_procs=self.num_procs)
		# self._session.run([init_op])
		self.next_elem = self.iterator.get_next()
	"""

	def get_activations(self, images=None, layer_names=None):
		return self._get_activations(layer_names)

	def _get_activations(self, layer_names):
		data = self._session.run(self._data_inputs.next_elem)
		layer_tensors = OrderedDict((layer, self._endpoints[
			layer if (layer != 'logits' or layer in self._endpoints) else next(reversed(self._endpoints))])
									for layer in layer_names)
		layer_outputs = self._session.run(layer_tensors, feed_dict={self._inputs: data[0]})
		return data[1], layer_outputs

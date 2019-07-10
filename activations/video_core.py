import copy
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from model_tools.activations.core import ActivationsExtractorHelper

from brainio_base.assemblies import NeuroidAssembly
from model_tools.activations.core import flatten

class VideoActivationsExtractorHelper(ActivationsExtractorHelper):

	def __init__(self, data_inputs=None, batch_size=60, *args, **kwargs):

		super(VideoActivationsExtractorHelper, self).__init__(batch_size=batch_size, *args, **kwargs)
		# TODO: remove datainputs dependencies
		self.data_inputs = data_inputs

	def _from_paths(self, layers, stimuli_paths):
		"""

		:param layers: layers to stimulate (type: list)
		:param stimuli_paths: paths to stimulus (type: list)
		:return:
		"""
		total = len(stimuli_paths) * self.data_inputs.fpv   # process length calculation for tqdm
		# create inputs from paths:
		self._inputs = self.data_inputs.make_from_paths(stimuli_paths) #TODO: remove datainputs dependencies
		# stimulate layer in batches (self._batch_size):
		stim_paths, layer_activations = self._get_activations_batched(total, layers, self._batch_size)
		# comnbine activation batches and return assembly:
		return self._package(layer_activations, stim_paths)

	def _get_activations_batched(self, path_len, layers, batch_size):
		"""
		Same as in model-tools, overrides call to self._get_batch_activations
		self.get_activations batched activations automatically
		:param path_len:
		:param layers:
		:param batch_size:
		:return:
		"""
		layer_activations = None
		stim_paths = []
		for batch_start in tqdm(range(0, path_len, batch_size), unit_scale=batch_size, desc="activations"):
			batch_end = min(batch_start + batch_size, path_len)
			self._logger.debug('Batch %d->%d/%d', batch_start, batch_end, path_len)
			batch_activations = self.get_activations(layer_names=layers)
			# TODO: remove get_stim_paths (remove datainputs dependencies)
			stim_paths += self.data_inputs.get_stim_paths()
			assert isinstance(batch_activations, OrderedDict)
			for hook in self._batch_activations_hooks.copy().values():  # copy to avoid handle re-enabling messing with the loop
				batch_activations = hook(batch_activations)

			if layer_activations is None:
				layer_activations = copy.copy(batch_activations)
			else:
				for layer_name, layer_output in batch_activations.items():
					layer_activations[layer_name] = np.concatenate((layer_activations[layer_name], layer_output))

		return stim_paths, layer_activations

	def _package_layer(self, layer_activations, layer, stimuli_paths):
		"""
		copy of _package_layer, added '4' as an allowed shape to account for temporal models in assert
		:param layer_activations:
		:param layer:
		:param stimuli_paths:
		:return:
		"""
		assert layer_activations.shape[0] == len(stimuli_paths)
		activations, flatten_indices = flatten(layer_activations, return_index=True)  # collapse for single neuroid dim
		assert flatten_indices.shape[1] in [1, 3, 4]  # either convolutional or fully-connected
		flatten_coord_names = ['channel', 'channel_x', 'channel_y']
		flatten_coords = {flatten_coord_names[i]: [sample_index[i] if i < flatten_indices.shape[1] else np.nan
		                                           for sample_index in flatten_indices]
		                  for i in range(len(flatten_coord_names))}
		layer_assembly = NeuroidAssembly(
			activations,
			coords={**{'stimulus_path': stimuli_paths,
			           'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
			           'model': ('neuroid', [self.identifier] * activations.shape[1]),
			           'layer': ('neuroid', [layer] * activations.shape[1]),
			           },
			        **{coord: ('neuroid', values) for coord, values in flatten_coords.items()}},
			dims=['stimulus_path', 'neuroid']
		)
		neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
			layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
		layer_assembly['neuroid_id'] = 'neuroid', neuroid_id
		return layer_assembly

	#TODO: def _package(self, layer_activations, stimuli_paths) may be a performance issue for very large datasets
	#TODO: Idea: merge activations after each batch in get_activations_batch

